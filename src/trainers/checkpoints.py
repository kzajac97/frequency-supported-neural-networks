import operator
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch.nn


def save_state(state, path: Path) -> None:
    """Saves torch tensor"""
    torch.save(state, path)


class AbstractCheckpointCreator(ABC):
    def __init__(self, model_save_path: Path):
        self.model_save_path = model_save_path

    @abstractmethod
    def save(self, model: torch.nn.Module, epoch: int, metrics: pd.Series):
        ...

    @abstractmethod
    def restore_model(self, model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        ...

    def restore_state(self) -> dict[str, Any]:
        # TODO: Add interface used by `SimpleStateCheckpoint` and other
        ...


class SimpleModelCheckpoint(AbstractCheckpointCreator):
    """Simple model checkpoint saving model state after each N epochs (specified by parameter)"""

    def __init__(self, model_save_path: Path, run_period: int):
        super().__init__(model_save_path)
        self.run_period = run_period
        self.restore_from = None

    @staticmethod
    def get_default_file_name(epoch: int) -> str:
        return f"simple_model_checkpoint_at_{epoch}.pt"

    def save(self, model: torch.nn.Module, epoch: int, metrics: pd.Series) -> None:
        """Saves model state to file under file name with epoch number"""
        if epoch % self.run_period == 0:
            target_path = self.model_save_path / Path(self.get_default_file_name(epoch))
            self.restore_from = target_path  # store last saved model path
            torch.save(model, target_path)

    def restore_model(self, model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        """Load model state to given model instance"""
        assert model is not None, "SimpleModelCheckpoint requires model instance to load_state_dict to!"
        assert self.restore_from is not None, "No model has be saved so far by this checkpoint!"

        return torch.load(self.restore_from)


class SimpleStateCheckpoint(AbstractCheckpointCreator):
    # TODO: Implement checkpoint for saving model and optimizer
    #       state to be able to resume training after crashes
    ...


class BestModelCheckpoint(AbstractCheckpointCreator):
    """
    Model checkpoint saving best seen models using one of
    available metrics to determine if they should be saved
    """

    def __init__(self, model_save_path: Path, metric_name: str, op: Callable[[float, float], bool] = operator.lt):
        """
        :param model_save_path:
        :param metric_name:
        :param op: python function which must return True for incoming metric, if the model is supposed to be saved
                   for minimization default is less_than (`operator.le`) operation
                   for maximization greater_than (`operator.ge`) should be used
        """
        super().__init__(model_save_path)
        self.metric_name = metric_name
        self.op = op

        self.restore_from = None
        self.metric_value = None

    @property
    def default_path(self) -> Path:
        """Default path under which checkpoint saves model states"""
        return self.model_save_path / Path("best_model_checkpoint.pt")

    def should_save(self, metrics: pd.Series) -> bool:
        """Function decides if incoming model should be saved, based on metrics it receives"""
        current_metric = metrics[self.metric_name]
        if self.metric_value is None:  # first save
            self.metric_value = current_metric
            return True
        elif self.op(current_metric, self.metric_value):
            self.metric_value = current_metric
            return True

        return False

    def save(self, model: torch.nn.Module, epoch: int, metrics: pd.Series):
        """Saves model state to file"""
        if self.should_save(metrics):
            self.metric_value = metrics[self.metric_name]
            torch.save(model, self.default_path)

    def restore_model(self, model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        """Load model state to given model instance"""
        assert model is not None, "SimpleModelCheckpoint requires model instance to load_state_dict to!"
        assert self.metric_value is not None, "No model has be saved so far by this checkpoint!"
        return torch.load(self.default_path)


class CheckpointHandler:
    """Abstract aggregator of checkpoints"""

    def __init__(self, checkpoints: list[AbstractCheckpointCreator], restore_from: Union[int, str]):
        """
        :param checkpoints: list of checkpoints
        :param restore_from: identifier of checkpoint to restore from, supports two options:
                             * int - i-th element of the checkpoints list is used to restore model after training
                             * str - model class with given name is used to restore model after training
        """
        if type(restore_from) is int:
            error_message = f"Invalid index given for checkpoint to restore from! {restore_from} >= {len(checkpoints)}"
            assert restore_from < len(checkpoints), error_message
        if type(restore_from) is str:
            possible_names = {checkpoint.__class__.__name__ for checkpoint in checkpoints}
            error_message = f"Trying to use invalid checkpoint to restore from! {restore_from} not in {possible_names}"
            assert restore_from in possible_names, error_message

        self.checkpoints = checkpoints
        self.restore_from = restore_from

    def save(self, model: torch.nn.Module, epoch: int, metrics: pd.Series) -> None:
        """Run save operation from each checkpoint in the group"""
        for checkpoint in self.checkpoints:
            checkpoint.save(model, epoch=epoch, metrics=metrics)

    def restore(self, model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        """
        Restores saved model from one of the checkpoints after training
        Some Checkpoints require passing model instance to load weights to
        """
        if type(self.restore_from) is int:
            return self.checkpoints[self.restore_from].restore_model(model)
        else:
            for checkpoint in self.checkpoints:
                if checkpoint.__class__.__name__ == self.restore_from:
                    return checkpoint.restore_model(model)
