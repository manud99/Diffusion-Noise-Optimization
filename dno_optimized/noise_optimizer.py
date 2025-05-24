import math
from typing import Callable, TypedDict

import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT
from torch_levenberg_marquardt.damping import StandardDampingStrategy
from tqdm import tqdm
from colorama import Fore

from dno_optimized.callbacks.callback import CallbackList
from dno_optimized.levenberg_marquardt import LevenbergMarquardt
from dno_optimized.options import DNOOptions, OptimizerType


def create_optimizer(
    optimizer: OptimizerType,
    params: ParamsT,
    config: DNOOptions,
    model: Callable[[Tensor], Tensor],
    criterion: Callable[[Tensor], Tensor],
) -> torch.optim.Optimizer:
    print("Config:", config)
    match optimizer:
        case OptimizerType.Adam:
            return torch.optim.Adam(params, lr=config.lr,
                                    betas=config.adam.betas,
                                    weight_decay=config.adam.weight_decay)
        case OptimizerType.LBFGS:
            return torch.optim.LBFGS(
                params,
                lr=config.lr,
                line_search_fn=config.lbfgs.line_search_fn,
                max_iter=config.lbfgs.max_iter,
                history_size=config.lbfgs.history_size,
            )
        case OptimizerType.SGD:
            return torch.optim.SGD(params, lr=config.lr)
        case OptimizerType.GaussNewton:
            raise NotImplementedError(optimizer)
        case OptimizerType.LevenbergMarquardt:
            assert config.levenbergMarquardt.solve_method in ['qr', 'cholesky', 'solve', 'lstsq']
            dampingStrategy = StandardDampingStrategy(
                starting_value=config.levenbergMarquardt.damping_strategy.starting_value,
                dec_factor=config.levenbergMarquardt.damping_strategy.dec_factor,
                inc_factor=config.levenbergMarquardt.damping_strategy.inc_factor,
                min_value=config.levenbergMarquardt.damping_strategy.min_value,
                max_value=config.levenbergMarquardt.damping_strategy.max_value,
                damping_mode=config.levenbergMarquardt.damping_strategy.damping_mode,
            )
            return LevenbergMarquardt(
                params,
                model=model,
                loss_fn=criterion,
                learning_rate=config.lr,
                damping_strategy=dampingStrategy,
                attempts_per_step=config.levenbergMarquardt.attempts_per_step,
                solve_method=config.levenbergMarquardt.solve_method,
            )
        case _:
            raise ValueError(f"`{optimizer}` is not a valid optimizer")


class DNOInfoDict(TypedDict):
    # Singleton values
    step: list[int]
    lr: list[float]
    perturb_scale: list[float]
    # Batched tensor values
    loss: torch.Tensor
    loss_objective: torch.Tensor
    loss_diff: torch.Tensor
    loss_decorrelate: torch.Tensor
    grad_norm: torch.Tensor
    diff_norm: torch.Tensor
    z: torch.Tensor
    x: torch.Tensor
    damping: list[float]
    num_attempts: list[int]


class DNOStateDict(TypedDict):
    z: torch.Tensor
    x: torch.Tensor
    hist: list[DNOInfoDict]
    stop_optimize: int


def default_info() -> DNOInfoDict:
    return {
        "step": [],
        "lr": [],
        "perturb_scale": [],
        "loss": torch.empty([]),
        "loss_objective": torch.empty([]),
        "loss_diff": torch.empty([]),
        "loss_decorrelate": torch.empty([]),
        "grad_norm": torch.empty([]),
        "diff_norm": torch.empty([]),
        "x": torch.empty([]),
        "z": torch.empty([]),
        "damping": [],
        "num_attempts": [],
    }


class DNO:
    """
    Args:
        start_z: (N, 263, 1, 120)
    """

    def __init__(
        self,
        model,
        criterion: Callable[[Tensor], Tensor],
        start_z: Tensor,
        conf: DNOOptions,
        callbacks: "CallbackList | None" = None,
    ):
        self.model = model
        self.criterion = criterion
        # for diff penalty
        self.start_z = start_z.detach()
        self.conf = conf

        self.current_z = self.start_z.clone().requires_grad_(True)
        # excluding the first dimension (batch size)
        self.dims = list(range(1, len(self.start_z.shape)))

        self.optimizer = create_optimizer(self.conf.optimizer, [self.current_z], self.conf, model, self.compute_raw_loss)
        print(f"INFO: Using {self.conf.optimizer.name} optimizer with LR of {self.conf.lr:.2g}")

        self.lr_scheduler = []
        if conf.lr_warm_up_steps > 0:
            self.lr_scheduler.append(lambda step: warmup_scheduler(step, conf.lr_warm_up_steps))
            print(f"INFO: Using linear learning rate warmup over {conf.lr_warm_up_steps} steps")
        if conf.lr_decay_steps > 0:
            self.lr_scheduler.append(
                lambda step: cosine_decay_scheduler(step, conf.lr_decay_steps, conf.num_opt_steps, decay_first=False)
            )
            print(f"INFO: Using cosine learning rate decay over {conf.lr_decay_steps} steps")

        print(f"INFO: Gradient normalization: {'ON' if self.conf.normalize_gradient else 'OFF'}")
        if self.conf.gradient_clip_val is not None:
            print(f"INFO: Gradient clip value: {self.conf.gradient_clip_val:.3f}")

        self.step_count = 0

        # Optimizer closure running variables
        self.last_x: torch.Tensor | None = None
        self.lr_frac: float | None = None

        # history of the optimization (for each step and each instance in the batch)
        # hist = {
        #    "step": [step] * batch_size,
        #    "lr": [lr] * batch_size,
        #    ...
        # }
        self.hist: list[DNOInfoDict] = []
        self.info: DNOInfoDict = {}  # type: ignore

        self.callbacks = callbacks or CallbackList()
        print("INFO: Using the following callbacks:")
        print(*[f"- {cb}" for cb in self.callbacks], sep="\n")
        print("====================================\n")

    @property
    def batch_size(self):
        return self.start_z.size(0)

    def __call__(self, num_steps: int | None = None):
        return self.optimize(num_steps=num_steps)

    def optimize(self, num_steps: int | None = None):
        if num_steps is None:
            num_steps = self.conf.num_opt_steps

        batch_size = self.batch_size

        self.step_count = 0
        self.callbacks.invoke(self, "train_begin", num_steps=num_steps, batch_size=batch_size)

        pb = tqdm(total=num_steps)
        x: torch.Tensor = torch.empty([])  # Will be initialized in training loop
        for i in range(num_steps):
            self.step_count += 1

            def closure():
                nonlocal x
                # Reset gradients
                self.optimizer.zero_grad()
                # Compute output based on current noise
                x = self.model(self.current_z)
                # Single step forward and backward
                loss = self.compute_loss(x, batch_size=batch_size)
                return loss.item()

            # Pre-step callbacks
            res = self.callbacks.invoke(self, "step_begin", pb=pb, step=self.step_count)
            if res.stop:
                break

            # Step optimization and add noise after optimization step
            self.optimizer.step(closure)
            self.last_x = x
            self.lr_frac = self.step_schedulers(batch_size=batch_size)
            self.noise_perturbation(self.lr_frac, batch_size=batch_size)

            # Merge logs from LevenbergMarquardt with our info object
            stop_training = False
            if isinstance(self.optimizer, LevenbergMarquardt):
                logs = self.optimizer.logs
                self.info['damping'] = [logs['damping']] * batch_size
                self.info['num_attempts'] = [logs['attempts']] * batch_size
                if logs['stop_training']:
                    pb.write(Fore.YELLOW + "WARNING: LevenbergMarquardt suggests to stop the training now!" + Fore.RESET)
                    stop_training = True

            self.update_metrics(x)

            # Post-step callbacks
            res = self.callbacks.invoke(self, "step_end", pb=pb, step=self.step_count, info=self.info, hist=self.hist)
            if res.stop or stop_training:
                break

            pb.set_postfix({"loss": self.info["loss"].mean().item()})
            pb.update(1)

        pb.close()

        # Check for early stopping
        if self.step_count < num_steps:
            print(f"INFO: Stopping optimization early at step {self.step_count}/{num_steps}")

        hist = self.compute_hist(batch_size=batch_size)

        assert self.last_x is not None, "Missing result"
        state_dict = self.state_dict()

        self.callbacks.invoke(
            self, "train_end", num_steps=num_steps, batch_size=batch_size, hist=hist, state_dict=state_dict
        )

        print() # New line after end of optimization
        return state_dict

    def state_dict(self) -> DNOStateDict:
        hist = self.compute_hist(self.batch_size)
        assert self.last_x is not None, "Please run at least one optimization iteration before calling DNO.state_dict()"
        return {
            # Last step's z
            "z": self.current_z.detach(),
            # Previous step's x
            "x": self.last_x.detach(),
            "hist": hist,
            # Amount of performed optimize steps
            "stop_optimize": self.step_count,
        }

    def step_schedulers(self, batch_size: int):
        # learning rate scheduler
        lr_frac = 1
        if len(self.lr_scheduler) > 0:
            for scheduler in self.lr_scheduler:
                lr_frac *= scheduler(self.step_count)
            self.set_lr(self.conf.lr * lr_frac)
        self.info["lr"] = [self.conf.lr * lr_frac] * batch_size
        return lr_frac

    def set_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr
        if isinstance(self.optimizer, LevenbergMarquardt):
            self.optimizer.learning_rate = lr

    def compute_loss(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        self.info = default_info()
        self.info["step"] = [self.step_count] * batch_size

        loss = self.compute_raw_loss(x, batch_size)

        # Aggregate over batch: sum, mean, or other? e.g. max? min?
        # Original DNO: sum
        loss_agg = loss.sum()
        # backward
        loss_agg.backward()

        # log grad norm (before)
        assert self.current_z.grad is not None
        self.info["grad_norm"] = self.current_z.grad.norm(p=2, dim=self.dims).detach().cpu()

        # grad mode
        if self.conf.normalize_gradient:
            self.current_z.grad.data /= self.current_z.grad.norm(p=2, dim=self.dims, keepdim=True)

        # Gradient clipping
        if self.conf.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.current_z, self.conf.gradient_clip_val, norm_type=2)

        return loss_agg

    def compute_raw_loss(self, x: torch.Tensor, batch_size: int | None = None, store_info: bool = True):
        if batch_size is None:
            batch_size = self.batch_size

        loss = self.criterion(x)
        assert loss.shape == (batch_size,)

        # Clone necessary if already on CPU since we modify loss in-place
        if store_info: self.info["loss_objective"] = loss.detach().cpu().clone()

        # diff penalty
        loss_diff = (self.current_z - self.start_z).norm(p=2, dim=self.dims)
        assert loss_diff.shape == (batch_size,)
        if store_info: self.info["loss_diff"] = loss_diff.detach().cpu()
        if self.conf.diff_penalty_scale > 0:
            loss += self.conf.diff_penalty_scale * loss_diff

        # decorrelation loss
        loss_decorrelate = noise_regularize_1d(
            self.current_z,
            dim=self.conf.decorrelate_dim,
        )
        assert loss_decorrelate.shape == (batch_size,)
        if store_info: self.info["loss_decorrelate"] = loss_decorrelate.detach().cpu()
        if self.conf.decorrelate_scale > 0:
            loss += self.conf.decorrelate_scale * loss_decorrelate

        if store_info: self.info["loss"] = loss.detach().cpu().clone()
        return loss

    def noise_perturbation(self, lr_frac, batch_size):
        # noise perturbation
        # match the noise fraction to the learning rate fraction
        noise_frac = lr_frac
        self.info["perturb_scale"] = [self.conf.perturb_scale * noise_frac] * batch_size

        noise = torch.randn_like(self.current_z)
        self.current_z.data += noise * self.conf.perturb_scale * noise_frac

    def update_metrics(self, x):
        # log the norm(z - start_z)
        self.info["diff_norm"] = (self.current_z - self.start_z).norm(p=2, dim=self.dims).detach().cpu()

        # log current z
        self.info["z"] = self.current_z.detach().cpu()
        self.info["x"] = x.detach().cpu()

        self.hist.append(self.info)

    def compute_hist(self, batch_size):
        # output is a list (over batch) of dict (over keys) of lists (over steps)
        hist = []
        for i in range(batch_size):
            hist.append({})
            for k in self.hist[0].keys():
                hist[-1][k] = [info[k][i] for info in self.hist]
        return hist


def warmup_scheduler(step, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps
    return 1


def cosine_decay_scheduler(step, decay_steps, total_steps, decay_first=True):
    # decay the last "decay_steps" steps from 1 to 0 using cosine decay
    # if decay_first is True, then the first "decay_steps" steps will be decayed from 1 to 0
    # if decay_first is False, then the last "decay_steps" steps will be decayed from 1 to 0
    if step >= total_steps:
        return 0
    if decay_first:
        if step >= decay_steps:
            return 0
        return (math.cos((step) / decay_steps * math.pi) + 1) / 2
    else:
        if step < total_steps - decay_steps:
            return 1
        return (math.cos((step - (total_steps - decay_steps)) / decay_steps * math.pi) + 1) / 2


def noise_regularize_1d(noise, stop_at=2, dim=3):
    """
    Args:
        noise (torch.Tensor): (N, C, 1, size)
        stop_at (int): stop decorrelating when size is less than or equal to stop_at
        dim (int): the dimension to decorrelate
    """
    all_dims = set(range(len(noise.shape)))
    loss = 0
    size = noise.shape[dim]

    # pad noise in the size dimention so that it is the power of 2
    if size != 2 ** int(math.log2(size)):
        new_size = 2 ** int(math.log2(size) + 1)
        pad = new_size - size
        pad_shape = list(noise.shape)
        pad_shape[dim] = pad
        pad_noise = torch.randn(*pad_shape).to(noise.device)

        noise = torch.cat([noise, pad_noise], dim=dim)
        size = noise.shape[dim]

    while True:
        # this loss penalizes spatially correlated noise
        # the noise is rolled in the size direction and the dot product is taken
        # (bs, )
        loss = loss + (noise * torch.roll(noise, shifts=1, dims=dim)).mean(
            # average over all dimensions except 0 (batch)
            dim=list(all_dims - {0})
        ).pow(2)

        # stop when size is 8
        if size <= stop_at:
            break

        # (N, C, 1, size) -> (N, C, 1, size // 2, 2)
        noise_shape = list(noise.shape)
        noise_shape[dim] = size // 2
        noise_shape.insert(dim + 1, 2)
        noise = noise.reshape(noise_shape)
        # average pool over (2,) window
        noise = noise.mean([dim + 1])
        size //= 2

    return loss
