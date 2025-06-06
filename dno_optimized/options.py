"""
Dataclass schemas for all DNO experiment options. To be used with OmegaConf.structured(...)
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# Use enums instead of string literal types as OmegaConf does not support Literal['string1', ...]
# Also inherit from str and Enum to allow comparison to string:
#
# >>> DNOTask.trajectory_editing == "trajectory_editing"  # True
#
# This would return False if we didn't inherit from str.


class DNOTask(str, Enum):
    trajectory_editing = "trajectory_editing"
    pose_editing = "pose_editing"
    dense_optimization = "dense_optimization"
    motion_projection = "motion_projection"
    motion_blending = "motion_blending"
    motion_inbetweening = "motion_inbetweening"


class Dataset(str, Enum):
    kit = "kit"
    humanml = "humanml"
    humanct12 = "humanct12"
    uestc = "uestc"


class Arch(str, Enum):
    trans_enc = "trans_enc"
    trans_dec = "trans_dec"
    gru = "gru"


class DiffusionNoiseSchedule(str, Enum):
    linear = "linear"
    cosine = "cosine"


class OptimizerType(str, Enum):
    Adam = "Adam"
    LBFGS = "LBFGS"
    SGD = "SGD"
    GaussNewton = "GaussNewton"
    LevenbergMarquardt = "LevenbergMarquardt"


class LBFGSLineSearchFn(str, Enum):
    strong_wolfe = "strong_wolfe"


@dataclass
class LBFGSOptions:
    history_size: int = field(default=10, metadata={"help": "Update history size"})
    line_search_fn: LBFGSLineSearchFn | None = None
    max_iter: int = 20

@dataclass
class DampingStrategyOptions:
    starting_value: float = field(default=1e-3, metadata={"help": "Starting value for linear search"})
    dec_factor: float = field(default=0.1, metadata={"help": "Decrement factor on successful updates"})
    inc_factor: float = field(default=10, metadata={"help": "Increment factor on unsuccessful updates"})
    min_value: float = field(default=1e-10, metadata={"help": "Min damping factor"})
    max_value: float = field(default=1e10, metadata={"help": "Max damping factor"})
    damping_mode: str = field(default='standard', metadata={"help": "Damping strategy mode. Options: ['standard', 'adaptive', 'fletcher']"})
    conditional_stopping: bool = field(default=True, metadata={"help": "Conditional stopping condition"})

@dataclass
class LevenbergMarquardtOptions:
    attempts_per_step: int = field(
        default=10,
        metadata={"help": "Number of attempts per step (1 for editing, 2 for refinement, can go further for better results)"},
    )
    solve_method: str = field(
        default='qr',
        metadata={"help": "Method to solve the LSE constructed. Options: ['qr', 'cholesky', 'solve', 'lstsq']."},
    )
    loss_aggregation: str = field(
        default='sum',
        metadata={"help": "Method to aggregate many loss values into a single float. Options: ['sum', 'min', 'max']"}
    )
    damping_strategy: DampingStrategyOptions = field(default_factory=DampingStrategyOptions, metadata={"help": "Damping strategy options"})

@dataclass
class AdamOptions:
    weight_decay: float = field(
        default=0,
        metadata={
            "help": "weight decay"
        }
    )
    betas: tuple[float,float] = field(
        default=(0.9, 0.999),
        metadata={
            "help": "Beta range"
        }
    )

@dataclass
class DNOOptions:
    num_opt_steps: int = field(
        default=500,
        metadata={
            "help": "Number of optimization steps (300 for editing, 500 for refinement, can go further for better results)"
        },
    )
    lr: float = field(default=5e-2, metadata={"help": "Learning rate"})
    perturb_scale: float = field(default=0, metadata={"help": "scale of the noise perturbation"})
    diff_penalty_scale: float | None = field(
        default=None,
        metadata={"help": "penalty for the difference between the final z and the initial z. Default is 2e-3 for editing tasks (i.e. pose and trajectory editing) and 0 otherwise."},
    )
    lr_warm_up_steps: int = field(default=50, metadata={"help": "Number of warm-up steps for the learning rate"})
    lr_decay_steps: int = field(
        default=-1,
        metadata={"help": "Number of decay steps (if None, then set to num_opt_steps)"},
    )
    decay_first: bool = False
    decorrelate_scale: float = field(default=1000, metadata={"help": "penalty for the decorrelation of the noise"})
    decorrelate_dim: int = field(
        default=3,
        metadata={"help": "dimension to decorrelate (we usually decorrelate time dimension)"},
    )

    # Custom optimizer options
    optimizer: OptimizerType = field(default=OptimizerType.Adam, metadata={"help": "Optimizer to use for DNO."})
    lbfgs: LBFGSOptions = field(default_factory=LBFGSOptions, metadata={"help": "Options for LBFGS optimizer"})
    levenbergMarquardt: LevenbergMarquardtOptions = field(
        default_factory=LevenbergMarquardtOptions, metadata={"help": "Options for Levenberg Marquardt optimizer"}
    )
    adam: AdamOptions = field(default_factory=AdamOptions, metadata={"help": "Options for Adam optimizer"})

    enable_profiler: bool = field(default=False, metadata={"help": "Enable profiler"})

    normalize_gradient: bool = field(default=True, metadata={"help": "Enables gradient normalization during training."})
    gradient_clip_val: float | None = field(
        default=None,
        metadata={"help": "Sets the maximum 2-norm for gradients during optimization. Set to `None` to disable."},
    )

    def __post_init__(self):
        # if lr_decay_steps is not set, then set it to num_opt_steps
        if self.lr_decay_steps == -1:
            self.lr_decay_steps = self.num_opt_steps


@dataclass
class CallbackConfig:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class GenerateOptions:
    ## MISC
    num_dump_step: int = 1
    predict_xstart: bool = False

    ## GENERAL
    task: DNOTask = DNOTask.trajectory_editing
    model_path: str = "./save/mdm_avg_dno/model000500000_avg.pt"
    start_time: str = datetime.now().strftime("%y%m%d-%H%M%S")
    experiment: str | None = field(default=None, metadata={ "help": "Name of the experiment, which defined the folder name" })

    ## GENERATE
    text_prompt: str = "a person is jumping"
    motion_length: float = 6.0
    input_text: str = ""
    action_file: str = ""
    action_name: str = ""
    load_from: str = ""

    ## BASE
    cuda: bool = True
    device: int = 0
    seed: int = 20

    ## DATA
    dataset: Dataset = Dataset.humanml
    data_dir: str = ""
    dataloader_num_workers: int = 8

    ## MODEL
    arch: Arch = Arch.trans_enc
    emb_trans_dec: bool = False
    layers: int = 8
    latent_dim: int = 512
    cond_mask_prob: float = 0.1
    lambda_rcxyz: float = 0.0
    lambda_vel: float = 0.0
    lambda_fc: float = 0.0
    unconstrained: bool = False

    ## DIFFUSION
    noise_schedule: DiffusionNoiseSchedule = DiffusionNoiseSchedule.cosine
    diffusion_steps: int = 1000
    sigma_small: bool = True

    ## DNO TASK CONFIG
    use_obstacles: bool = False

    ## EXPERIMENT
    num_trials: int = 3
    num_ode_steps: int = 10
    gradient_checkpoint: bool = False
    use_ddim: bool = True
    max_frames: int = 196
    gen_batch_size: int = 1
    num_samples: int = 1
    num_repetitions: int = 1
    fps: float = 20.0
    guidance_param: float = 2.5

    dno: DNOOptions = field(default_factory=DNOOptions)

    callbacks: list[CallbackConfig] | None = field(
        default=None, metadata={"help": "Defines callbacks to use. If None, default callbacks are used."}
    )
    extra_callbacks: list[CallbackConfig] | None = field(
        default=None,
        metadata={
            "help": "Defines additional callbacks to be merged with default callbacks (or `callbacks`) with the `replace` strategy."
        },
    )

    def __post_init__(self):
        # Peform post-initialization work here
        pass

    ## COMPUTED PROPERTIES
    @property
    def niter(self):
        return re.match(r"model(.*)\.pt", os.path.basename(self.model_path)).group(1)  # type: ignore

    @property
    def n_frames(self):
        return min(self.max_frames, int(self.motion_length * self.fps))

    @property
    def gen_frames(self):
        # NOTE: Was hard-coded to 6.0 * fps before. Since motion_length is 6.0 by default, I assume that they should
        # match.
        return int(self.motion_length * self.fps)

    @property
    def batch_size(self):
        return self.num_samples

    @property
    def out_path(self):
        text_prompt_safe = re.sub(r"\s+", "_", self.text_prompt.strip())  # Trim and replace spaces by _
        text_prompt_safe = re.sub(r"[^\w\d_]", "", text_prompt_safe)  # Remove any non-word characters
        return (
            Path(self.model_path).parent
            / (self.experiment if self.experiment else f"samples_{self.niter}_seed{self.seed}_{text_prompt_safe}")
            / f"{self.task.name}_{self.start_time}_{self.dno.optimizer.name}"
        )
