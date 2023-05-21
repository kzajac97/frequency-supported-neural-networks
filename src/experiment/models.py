from types import MappingProxyType
from typing import Any

import torch.nn

from src.models.fmlp import FrequencyLinear, TimeFrequencyLinear
from src.models.linear import TimeSeriesLinear

ACTIVATIONS = MappingProxyType(
    {
        "elu": torch.nn.ELU,
        "identity": torch.nn.Identity,
        "hardshrink": torch.nn.Hardshrink,
        "hardsigmoid": torch.nn.Hardsigmoid,
        "hardtanh": torch.nn.Hardtanh,
        "hardswish": torch.nn.Hardswish,
        "leaky_relu": torch.nn.LeakyReLU,
        "log_sigmoid": torch.nn.LogSigmoid,
        "prelu": torch.nn.PReLU,
        "relu": torch.nn.ReLU,
        "relu6": torch.nn.ReLU6,
        "rrelu": torch.nn.RReLU,
        "selu": torch.nn.SELU,
        "celu": torch.nn.CELU,
        "gelu": torch.nn.GELU,
        "sigmoid": torch.nn.Sigmoid,
        "silu": torch.nn.SiLU,
        "mish": torch.nn.Mish,
        "softplus": torch.nn.Softplus,
        "softshrink": torch.nn.Softshrink,
        "softsign": torch.nn.Softsign,
        "tanh": torch.nn.Tanh,
        "tanhsrink": torch.nn.Tanhshrink,
        "threshold": torch.nn.Threshold,
        "GLU": torch.nn.GLU,
    }
)


def build_activation(name: str, parameters: dict[str, Any]):
    activation = ACTIVATIONS.get(name, None)
    assert (
        activation
    ), f"Attempting to use non-existing activation function! Available options are: {ACTIVATIONS.keys()}"
    return activation(**parameters)


def fmlp_from_parameters(parameters: dict[str, Any]) -> torch.nn.Module:
    layers = []

    layers.append(
        TimeFrequencyLinear(
            n_input_time_steps=parameters["n_input_time_steps"],
            n_output_time_steps=parameters["n_hidden_time_steps"],
            n_input_state_variables=parameters["n_input_state_variables"],
            n_output_state_variables=parameters["n_hidden_state_variables"],
        )
    )

    layers.append(
        build_activation(name=parameters["activation"], parameters=parameters.get("activation_parameters", {}))
    )

    for _ in range(parameters["n_hidden_layers"]):
        layers.append(
            TimeFrequencyLinear(
                n_input_time_steps=parameters["n_hidden_time_steps"],
                n_output_time_steps=parameters["n_hidden_time_steps"],
                n_input_state_variables=parameters["n_hidden_state_variables"],
                n_output_state_variables=parameters["n_hidden_state_variables"],
            )
        )

        layers.append(
            build_activation(name=parameters["activation"], parameters=parameters.get("activation_parameters", {}))
        )

    layers.append(
        TimeFrequencyLinear(
            n_input_time_steps=parameters["n_hidden_time_steps"],
            n_output_time_steps=parameters["n_output_time_steps"],
            n_input_state_variables=parameters["n_hidden_state_variables"],
            n_output_state_variables=parameters["n_output_state_variables"],
        )
    )

    return torch.nn.Sequential(*layers)


def time_series_mlp_from_parameters(parameters: dict[str, Any]) -> torch.nn.Module:
    layers = []

    layers.append(
        TimeSeriesLinear(
            n_input_time_steps=parameters["n_input_time_steps"],
            n_output_time_steps=parameters["n_hidden_time_steps"],
            n_input_state_variables=parameters["n_input_state_variables"],
            n_output_state_variables=parameters["n_hidden_state_variables"],
        )
    )

    layers.append(
        build_activation(name=parameters["activation"], parameters=parameters.get("activation_parameters", {}))
    )

    for _ in range(parameters["n_hidden_layers"]):
        layers.append(
            TimeSeriesLinear(
                n_input_time_steps=parameters["n_hidden_time_steps"],
                n_output_time_steps=parameters["n_hidden_time_steps"],
                n_input_state_variables=parameters["n_hidden_state_variables"],
                n_output_state_variables=parameters["n_hidden_state_variables"],
            )
        )

        layers.append(
            build_activation(name=parameters["activation"], parameters=parameters.get("activation_parameters", {}))
        )

    layers.append(
        TimeSeriesLinear(
            n_input_time_steps=parameters["n_hidden_time_steps"],
            n_output_time_steps=parameters["n_output_time_steps"],
            n_input_state_variables=parameters["n_hidden_state_variables"],
            n_output_state_variables=parameters["n_output_state_variables"],
        )
    )

    return torch.nn.Sequential(*layers)


def frequency_only_mlp_from_parameters(parameters: dict[str, Any]) -> torch.nn.Module:
    layers = []

    layers.append(
        FrequencyLinear(
            n_input_time_steps=parameters["n_input_time_steps"],
            n_output_time_steps=parameters["n_hidden_time_steps"],
            n_input_state_variables=parameters["n_input_state_variables"],
            n_output_state_variables=parameters["n_hidden_state_variables"],
        )
    )

    layers.append(
        build_activation(name=parameters["activation"], parameters=parameters.get("activation_parameters", {}))
    )

    for _ in range(parameters["n_hidden_layers"]):
        layers.append(
            FrequencyLinear(
                n_input_time_steps=parameters["n_hidden_time_steps"],
                n_output_time_steps=parameters["n_hidden_time_steps"],
                n_input_state_variables=parameters["n_hidden_state_variables"],
                n_output_state_variables=parameters["n_hidden_state_variables"],
            )
        )

        layers.append(
            build_activation(name=parameters["activation"], parameters=parameters.get("activation_parameters", {}))
        )

    layers.append(
        FrequencyLinear(
            n_input_time_steps=parameters["n_hidden_time_steps"],
            n_output_time_steps=parameters["n_output_time_steps"],
            n_input_state_variables=parameters["n_hidden_state_variables"],
            n_output_state_variables=parameters["n_output_state_variables"],
        )
    )

    return torch.nn.Sequential(*layers)
