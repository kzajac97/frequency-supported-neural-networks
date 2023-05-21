from types import MappingProxyType
from typing import Any, Union

import torch

from src.trainers import callbacks, checkpoints

TORCH_OPTIMIZERS = MappingProxyType(
    {
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sparse_adam": torch.optim.SparseAdam,
        "adamax": torch.optim.Adamax,
        "asgd": torch.optim.ASGD,
        "lbfgs": torch.optim.LBFGS,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
        "sgd": torch.optim.SGD,
    }
)


TORCH_LOSS_FUNCTIONS = MappingProxyType(
    {
        "mse": torch.nn.MSELoss,
        "l1": torch.nn.L1Loss,
        "huber": torch.nn.HuberLoss,
        "smooth_l1": torch.nn.SmoothL1Loss,
    }
)

CALLBACKS = MappingProxyType(
    {
        "early_stopping": callbacks.EarlyStoppingCallback,
        "regression_report": callbacks.RegressionReportCallback,
        "wandb_regression_report": callbacks.WandbRegressionReportCallback,
        "training_timeout": callbacks.TrainingTimeoutCallback,
        "training_history": callbacks.TrainingHistoryWriterCallback,
    }
)

CHECKPOINTS = MappingProxyType(
    {
        "simple_model_checkpoint": checkpoints.SimpleModelCheckpoint,
        "best_model_checkpoint": checkpoints.BestModelCheckpoint,
    }
)


def build_optimizer(model: torch.nn.Module, name: str, parameters: dict[str, Any]) -> torch.optim.Optimizer:
    """Builds torch optimizer from available optimizer classes using string accessor"""
    optimizer = TORCH_OPTIMIZERS.get(name)
    assert optimizer, f"Attempting to use non-existing  optimizer! Valid parameters are: {TORCH_OPTIMIZERS.keys()}"
    return optimizer(model.parameters(), **parameters)


def build_loss_function(name, parameters: dict[str, Any]) -> torch.nn.modules.loss._Loss:
    """
    Builds torch loss function from available loss classes using string accessor
    Supports only regression losses!
    """
    loss_fn = TORCH_LOSS_FUNCTIONS.get(name)
    assert loss_fn, f"Attempting to use non-existing loss function! Valid parameters are: {TORCH_LOSS_FUNCTIONS.keys()}"
    return loss_fn(**parameters)


def build_callback_handler(names: list[str], parameters: list[dict]) -> callbacks.CallbackHandler:
    """
    :param names: list of callback names
    :param parameters: dict of name to callback parameters for given names

    :return: list of build callbacks
    """
    callbacks_list = []

    for name, parameter_set in zip(names, parameters):
        callback = CALLBACKS.get(name)
        assert callback, f"Attempting to use non-existing callback! Supported parameters are: {CALLBACKS.keys()}"
        callbacks_list.append(callback(**parameter_set))

    return callbacks.CallbackHandler(callbacks_list)


def build_checkpoint_list(names: list[str], parameters: list[dict], restore_from: Union[str, int]):
    """
    :param names: list of callback names
    :param parameters: dict of name to callback parameters for given names
    :param restore_from: identifier of checkpoint to restore from, supports two options:
                         * int - i-th element of the checkpoints list is used to restore model after training
                         * str - model class with given name is used to restore model after training

    :return: list of build callbacks
    """
    checkpoint_list = []

    for name, parameter_set in zip(names, parameters):
        checkpoint = CHECKPOINTS.get(name)
        assert checkpoint, f"Attempting to use non-existing callback! Supported parameters are: {CHECKPOINTS.keys()}"
        checkpoint_list.append(checkpoint(**parameter_set))

    return checkpoints.CheckpointHandler(checkpoint_list, restore_from=restore_from)
