import torch


class TimeSeriesLinear(torch.nn.Module):
    """
    Linear layer adjusted to process time multi-dimensional series data
    It processes input with shape (B, Ti, Si) and returns (B, To, So), where:
        * B - batch
        * Ti/To - input/output time steps
        * Si/So - input/output state dimensions

    For single dimensional systems, pass (B, T, 1)
    Using this module with such input will be equivalent to performing: squeeze, linear, unsqueeze
    """

    def __init__(
        self,
        n_input_time_steps: int,
        n_output_time_steps: int,
        n_input_state_variables: int,
        n_output_state_variables: int,
        use_bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param n_input_time_steps: number of time steps passed in input
        :param n_output_time_steps: number of produces time steps
        :param n_input_state_variables: number of state variables in input
        :param n_output_state_variables: number of state variables in output
        :param use_bias: additional param for torch.nn.Linear, defaults to True
        :param dtype: data type, defaults to torch.float32
        """
        super(TimeSeriesLinear, self).__init__()

        self.n_input_time_steps = n_input_time_steps
        self.n_output_time_steps = n_output_time_steps
        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables
        self.dtype = dtype

        self.linear = torch.nn.Linear(
            in_features=self.n_input_time_steps * self.n_input_state_variables,
            out_features=self.n_output_time_steps * self.n_output_state_variables,
            bias=use_bias,
            dtype=dtype,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]

        variables = inputs.to(self.dtype).flatten(start_dim=1)
        variables = self.linear(variables)
        outputs = variables.reshape((batch_size, self.n_output_time_steps, self.n_output_state_variables))

        return outputs
