from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch.utils.data

from src.datasets.noise import AbstractNoiseGenerator
from src.datasets.sequences import generate_time_series_windows, time_series_train_test_split
from src.utils.iterables import filter_dict


class AbstractDataLoader(ABC):
    """Interface for DataLoader"""

    @property
    @abstractmethod
    def training_data_loader(self) -> Iterable:
        ...

    @property
    @abstractmethod
    def validation_data_loader(self) -> Iterable:
        ...

    @property
    @abstractmethod
    def test_data_loader(self) -> Iterable:
        ...


class TorchTimeSeriesCsvDataLoader(AbstractDataLoader):
    def __init__(
        self,
        dataset_path: Path,
        input_columns: list[str],
        output_columns: list[str],
        window_generation_config: dict[str, int],
        test_size: Union[int, float] = 0.5,
        validation_size: Optional[Union[int, float]] = None,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param dataset_path: path to CSV file with dataset
        :param input_columns: names of columns containing systems inputs
        :param output_columns: names of columns containing systems outputs
        :param window_generation_config: dict with window generation configuration, it can contain empty values
                                         which will results in using defaults from `generate_time_series_windows`
        :param test_size: test size in samples or ration, for details see `time_series_train_test_split`
        :param validation_size: size of validation dataset as a ratio of training data or absolute number of samples
                                validation samples are drawn from training data
        :param batch_size: batch size used to train and test the model by torch DataLoaders
        :param dtype: tensor data type, must much model data type in experiment, defaults to torch.float32
        """
        self.dataset_path = dataset_path
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.test_size = test_size
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.window_generation_config = window_generation_config

    @cached_property
    def dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset_path)

    def get_train_and_test_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Pre-loaded dataset split into train and test"""
        return time_series_train_test_split(self.dataset, test_size=self.test_size)  # type: ignore

    @property
    def train_dataset(self) -> pd.DataFrame:
        train_dataset, _ = self.get_train_and_test_datasets()
        return train_dataset

    @property
    def test_dataset(self) -> pd.DataFrame:
        _, test_dataset = self.get_train_and_test_datasets()
        return test_dataset

    @property
    def train_windows(self):
        """Caches time-series windows for training data"""
        inputs = self.train_dataset[self.input_columns].values if self.input_columns else None
        outputs = self.train_dataset[self.output_columns].values

        return self.generate_windows(inputs, outputs)

    def compute_n_validation_samples(self, n_windows: int) -> int:
        """Computes the number of validation samples based on the number of training windows"""
        if self.validation_size > 1:
            error_message = "Number of validation windows  must be smaller than total number of windows!"
            assert self.validation_size < n_windows, error_message + f"{self.validation_size} >= {n_windows}"
            return int(self.validation_size)
        else:
            return int(self.validation_size * n_windows)

    def get_validation_indices(self, n_windows: int) -> None:
        """Returns randomly chosen validation samples"""
        n_validation_samples = self.compute_n_validation_samples(n_windows)
        return np.random.choice(list(range(n_windows)), size=n_validation_samples)

    def generate_windows(self, inputs: Optional[Sequence], outputs: Sequence) -> dict[str, Sequence]:
        """
        Use generate_time_series_windows with given configuration to generate windows used by the model

        Always generates using order convention, which must be handled by the model:
            * backward_inputs
            * backward_outputs
            * forward_inputs
            * forward_outputs
        """
        parameters = dict(
            outputs=outputs,
            inputs=inputs,
            shift=self.window_generation_config.get("shift"),
            forward_input_window_size=self.window_generation_config.get("forward_input_window_size"),
            forward_output_window_size=self.window_generation_config.get("forward_output_window_size"),
            backward_input_window_size=self.window_generation_config.get("backward_input_window_size"),
            backward_output_window_size=self.window_generation_config.get("backward_output_window_size"),
            forward_input_mask=self.window_generation_config.get("forward_input_mask"),
            forward_output_mask=self.window_generation_config.get("forward_output_mask"),
            backward_input_mask=self.window_generation_config.get("backward_input_mask"),
            backward_output_mask=self.window_generation_config.get("backward_output_mask"),
        )

        return generate_time_series_windows(**filter_dict(None, parameters))

    def window_to_tensor(
        self, window: np.array, select: Optional[Sequence] = None, delete: Optional[Sequence] = None
    ) -> torch.Tensor:
        """
        Function converts numpy array with time series window into high dimensional tensor

        :param window: array with time-window slice of sequential data
        :param select: array to select into tensor, if None given are all selected
        :param delete: array indices to remove from tensor, if None given nothing is removed

        :return: pytorch tensor with window array content
        """
        if select is not None and delete is not None:
            raise ValueError("Cannot use select and delete at the same time!")
        if select is not None:
            window = np.take(window, select, axis=0)
        if delete is not None:
            window = np.delete(window, delete, axis=0)

        return torch.from_numpy(window).to(self.dtype)

    @staticmethod
    def tensors_to_data_loader(tensors: list[torch.Tensor], **kwargs) -> Iterable:
        """Converts collection of tensors to DataLoader object"""
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), **kwargs)

    def arrays_to_tensors(
        self, arrays: list[np.array], select: Optional[Sequence] = None, delete: Optional[Sequence] = None
    ) -> list[torch.Tensor]:
        """Converts list of arrays into list of tensors"""
        return [self.window_to_tensor(window, select=select, delete=delete) for window in arrays if window.size > 0]

    def get_training_and_validation_data_loader(self) -> Tuple[Iterable, Optional[Iterable]]:
        """
        :return: pytorch DataLoader with train and validation data
                 if `self.validation_size` is not given validation_data_loader is None
        """
        if self.validation_size:
            validation_indices = self.get_validation_indices(len(self.train_windows["forward_inputs"]))

            training_tensors = self.arrays_to_tensors(list(self.train_windows.values()), delete=validation_indices)
            validation_tensors = self.arrays_to_tensors(list(self.train_windows.values()), select=validation_indices)

            return (
                self.tensors_to_data_loader(training_tensors, batch_size=self.batch_size, shuffle=True),
                self.tensors_to_data_loader(validation_tensors, batch_size=self.batch_size, shuffle=True),
            )

        tensors = [self.window_to_tensor(window) for window in self.train_windows.values() if len(window) != 0]
        return self.tensors_to_data_loader(tensors, batch_size=self.batch_size, shuffle=True), None

    @cached_property
    def training_data_loader(self) -> Iterable:
        """Generates training data and returns torch DataLoader"""
        training_loader, _ = self.get_training_and_validation_data_loader()
        return training_loader

    @cached_property
    def validation_data_loader(self) -> Iterable:
        """
        Generates training data and returns torch DataLoader
        with validation data which is removed from training data loader
        """
        _, validation_loader = self.get_training_and_validation_data_loader()

        if not validation_loader:
            raise ValueError("Cannot use validation_loader if `validation_size` is not given!")

        return validation_loader

    @cached_property
    def test_data_loader(self) -> Iterable:
        """Generates training data and returns torch DataLoader"""
        inputs = self.test_dataset[self.input_columns].values if self.input_columns else None
        outputs = self.test_dataset[self.output_columns].values

        test_windows = self.generate_windows(inputs, outputs)
        tensors = self.arrays_to_tensors(list(test_windows.values()))

        return self.tensors_to_data_loader(tensors, batch_size=self.batch_size, shuffle=False)


class NoisyTorchTimeSeriesCsvDataLoader(TorchTimeSeriesCsvDataLoader):
    """
    Class implemented dataset adding controlled noise to experimental measurements using AbstractNoiseGenerator
    subclasses, for parameter description see TorchTimeSeriesCsvDataLoader.__doc__

    This class will only result in deterministic results if noise generators cache the data or use deterministic noise
    (for example stored in a file). Using random noise can controlled using random seeds, but when used with simple
    random number generators the datasets produced will not non-deterministic and different with each experiment
    """

    def __init__(
        self,
        dataset_path: Path,
        input_columns: list[str],
        output_columns: list[str],
        window_generation_config: dict[str, int],
        input_noise_generator: AbstractNoiseGenerator,
        output_noise_generator: AbstractNoiseGenerator,
        test_size: Union[int, float] = 0.5,
        validation_size: Optional[Union[int, float]] = None,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param input_noise_generator: implementation of AbstractNoiseGenerator interface adding correlated or
                                      uncorrelated noise to system inputs
        :param output_noise_generator: analogical object to input_noise_generator, but applied to output
        """
        self.input_noise_generator = input_noise_generator
        self.output_noise_generator = output_noise_generator

        super().__init__(
            dataset_path=dataset_path,
            input_columns=input_columns,
            output_columns=output_columns,
            window_generation_config=window_generation_config,
            test_size=test_size,
            validation_size=validation_size,
            batch_size=batch_size,
            dtype=dtype,
        )

    @cached_property
    def dataset(self) -> pd.DataFrame:
        """Overloads TorchTimeSeriesCsvDataLoader dataset property with dataset with noise added to measurements"""
        dataset = pd.read_csv(self.dataset_path)
        dataset[self.input_columns] = self.input_noise_generator(dataset[self.input_columns])
        dataset[self.output_columns] = self.output_noise_generator(dataset[self.output_columns])

        return dataset
