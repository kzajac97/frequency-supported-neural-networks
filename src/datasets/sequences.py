from typing import Optional, Sequence, Union

import numpy as np


def time_series_train_test_split(sequence: Sequence, test_size: Union[int, float]) -> tuple[Sequence, Sequence]:
    """
    Splits time series into test and train set

    :param sequence: sequence containing time series measurements
    :param test_size: test size, given as fraction or number of samples

    :return: two sequences containing train and test data respectively
    """
    first_test_index = len(sequence) - test_size if test_size > 1 else int((1 - test_size) * len(sequence))
    return sequence[:first_test_index], sequence[first_test_index:]


def generate_time_series_windows(
    outputs: np.array,
    inputs: Optional[np.array] = None,
    *,
    forward_input_window_size: int = 0,
    backward_input_window_size: int = 0,
    forward_output_window_size: int = 0,
    backward_output_window_size: int = 0,
    shift: int = 1,
    forward_input_mask: int = 0,
    backward_input_mask: int = 0,
    forward_output_mask: int = 0,
    backward_output_mask: int = 0,
):
    """
    Utility function for generating input-output pairs for dynamical system identification algorithms.
    It supports generating inputs for both simulation and predictive modelling,
    Definitions of those in dynamical system identification can be found in https://arxiv.org/abs/1902.00683

    This function assumes 2D inputs with shape (TIME_STEPS, STATE_DIMENSIONS)
    and generates 3D outputs with shape (BATCH, TIME_STEPS, STATE_DIMENSIONS)

    It does not do any validation of its arguments, which should be handled by interfaces for different modes
    of sequences slicing, for more details see `src.utils.datasets.interfaces.py`

    :param outputs: array of outputs to the dynamical system, time-series of system state measurements
    :param inputs: array of inputs to the dynamical system, time-series of forcing measurements
    :param forward_input_window_size: number of input (forcing) measurements to include forward from prediction start
    :param backward_input_window_size: number of input (forcing)  measurements to include forward from prediction start
    :param forward_output_window_size: number of output (state) measurements to include forward from prediction start
    :param backward_output_window_size: number of output (state) measurements to include before the prediction start
    :param shift: number of samples to move the prediction starting point, can generate overlapping samples
    :param forward_input_mask: number of masked samples for forward inputs (forcing)
    :param backward_input_mask: number of masked samples for backward inputs (forcing)
    :param forward_output_mask: number of masked samples for forward outputs (states)
    :param backward_output_mask: number of masked samples for backward outputs (states)

    :return: dict with numpy array of generated time-series slices
    """
    sequences = {
        "backward_inputs": [],
        "backward_outputs": [],
        "forward_inputs": [],
        "forward_outputs": [],
    }

    def sample_forward(values: np.array, current_index: int, window_size: int, mask: int):
        return values[current_index + mask : current_index + window_size, :]  # noqa

    def sample_backward(values: np.array, current_index: int, window_size: int, mask: int):
        return values[current_index - window_size : current_index - mask, :]  # noqa

    max_backward_size = max(backward_input_window_size, backward_output_window_size)
    for index in list(range(max_backward_size, len(outputs)))[::shift]:
        if index + forward_output_window_size > len(outputs):
            break  # break when window size longer than input

        # fmt: off
        if forward_input_window_size > 0:
            samples = sample_forward(inputs, index, window_size=forward_input_window_size, mask=forward_input_mask)
            sequences["forward_inputs"].append(samples)
        if backward_input_window_size > 0:
            samples = sample_backward(inputs, index, window_size=backward_input_window_size, mask=backward_input_mask)
            sequences["backward_inputs"].append(samples)
        if forward_output_window_size > 0:
            samples = sample_forward(outputs, index, window_size=forward_output_window_size, mask=forward_output_mask)
            sequences["forward_outputs"].append(samples)
        if backward_output_window_size > 0:
            samples = sample_backward(outputs, index, window_size=backward_output_window_size, mask=backward_output_mask)  # noqa
            sequences["backward_outputs"].append(samples)
        # fmt: on

    return {key: np.asarray(values) for key, values in sequences.items()}
