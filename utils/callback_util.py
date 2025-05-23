from dno_optimized.callbacks.callback import Callback, CallbackList
from dno_optimized.callbacks.early_stopping import EarlyStoppingCallback
from dno_optimized.callbacks.generate_video import GenerateVideoCallback
from dno_optimized.callbacks.profiler import ProfilerCallback
from dno_optimized.callbacks.save_top_k import SaveTopKCallback
from dno_optimized.callbacks.tensorboard import TensorboardCallback
from dno_optimized.options import CallbackConfig, GenerateOptions


def create_callback(name: str, options: GenerateOptions, callback_args: dict) -> Callback:
    """Creates a callback from global options and callback configuration.

    :param name: Name of the callback to instantiate.
    :param options: Global generate options
    :param callback_args: Callback options. This is passed to `Callback.from_config(...)`.
    :raises KeyError: The provided callback is not registered.
    :return: Instantiated callback
    """
    callback = None
    match name:
        case "tensorboard":
            callback = TensorboardCallback.from_config(options, callback_args)
        case "early_stopping":
            callback = EarlyStoppingCallback.from_config(options, callback_args)
        case "save_top_k":
            callback = SaveTopKCallback.from_config(options, callback_args)
        case "profiler":
            callback = ProfilerCallback.from_config(options, callback_args)
        case "generate_video":
            callback = GenerateVideoCallback.from_config(options, callback_args)
        case _:
            raise KeyError(f"`{name}` is not a valid callback name")
    # Set common options
    callback.init_from_config(options, callback_args)
    return callback


def default_callbacks(options: GenerateOptions, run_post_init: bool = False) -> CallbackList:
    """Get a list of default callbacks.

    :param options: Global generate options
    :param run_post_init: Whether post_init should be ran on the resulting callback list, defaults to True
    :return: Callback list
    """
    cb_list = CallbackList(
        [
            TensorboardCallback.from_config(options, {"every_n_steps": 10}),
            EarlyStoppingCallback.from_config(options, {"patience": 50, "min_improvement": 1e-4, "abs_value": 1e-5}),
            SaveTopKCallback.from_config(options, {}),
        ]
    )
    if run_post_init:
        cb_list.post_init()
    return cb_list


def callback_list_from_config(configs: list[CallbackConfig], options: GenerateOptions, run_post_init: bool = False):
    """Create a callback list from a list of callback configurations

    :param configs: List of callback configs
    :param options: Global generate options
    :param run_post_init: Whether to run post_init on the resulting callback list, defaults to True
    :return: Callback list
    """
    cb_list = CallbackList([create_callback(conf.name, options, conf.args) for conf in configs])
    if run_post_init:
        cb_list.post_init()
    return cb_list


def callbacks_from_options(options: GenerateOptions, post_init_kwargs: dict | None = None):
    """Instantiate all callbacks using `callbacks` (if present), else default callbacks, and merging with `extra_callbacks`.

    :param options: Global generate options
    :return: Callback list
    """
    callbacks = (
        callback_list_from_config(options.callbacks, options) if options.callbacks else default_callbacks(options)
    )
    if options.extra_callbacks:
        callbacks.extend(callback_list_from_config(options.extra_callbacks, options), mode="replace")
    callbacks.post_init(**(post_init_kwargs or {}))
    return callbacks
