import warnings
from pathlib import Path
from typing import Callable, Self, override

from dno_optimized.callbacks.callback import CallbackList, CallbackStepAction
from dno_optimized.noise_optimizer import DNOInfoDict
from dno_optimized.options import GenerateOptions

from .callback import Callback


class GenerateVideoCallback(Callback):
    def __init__(self, out_dir: str = "videos", every_n_steps: int | None = None, start_after: int | None = None):
        super().__init__(every_n_steps, start_after)

        self.out_dir = Path(out_dir)

        self.process_fn: Callable | None = None

    def __post_init__(self, callbacks: CallbackList, process_fn: Callable):
        # Get the result processing function from main function
        self.process_fn = process_fn

    @override
    @classmethod
    def from_config(cls, options: GenerateOptions, config: dict) -> Self:
        return cls(
            out_dir=config.get("out_dir", options.out_path / "intermediate_videos"),
            every_n_steps=config.get("every_n_steps", 10),  # Override default
        )

    def on_step_end(self, step: int, info: DNOInfoDict, hist: list[DNOInfoDict]) -> CallbackStepAction | None:
        if self.process_fn is None:
            warnings.warn(
                "Using GenerateVideo callback, but process_fn was not passed. "
                "Please pass the output processing function to callbacks.post_init(...)"
            )
            return

        self.progress.write("Saving intermedaite video")

        # Get current output
        out = self.dno.state_dict()

        # Visualize
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.process_fn(out, save=False, plots=False, videos=True, step=step, out_dir=str(self.out_dir))
