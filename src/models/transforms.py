from typing import Optional

import torch


class RealForwardFourier(torch.nn.Module):
    """
    Module computes Fourier Transform for real input
    It is used as a building block of FMLP and FLSTM networks

    It processes tensors with shape (B, T, S), where
        * B - batch
        * T - time steps
        * S - state dimensions

    For processing inputs of system with single state dimension pass Tensor with shape: (B, T, 1) (unsqueezed tensor)
    """

    def __init__(
        self, n_time_steps: Optional[int] = None, norm: Optional[str] = None, dtype: torch.dtype = torch.cfloat
    ):
        """
        :param n_time_steps: number of time to produce steps, see torch.fft.rfft for details
        :param norm: norm of RFFT, see torch.fft.rfft for details
        :param dtype: output data type, defaults to cfloat and should be complex
                      otherwise will cause loss of information after the RFFT transform
        """
        super(RealForwardFourier, self).__init__()

        self.n_input_time_steps = n_time_steps
        self.norm = norm
        self.dtype = dtype

        self.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.fft.rfft(inputs, n=self.n_input_time_steps, norm=self.norm, dim=1)
        return outputs.to(self.dtype)


class ComplexForwardFourier(torch.nn.Module):
    """
    Module computes Fourier Transform for any input
    It is used as a building block of FMLP and FLSTM networks

    It processes tensors with shape (B, T, S), where
        * B - batch
        * T - time steps
        * S - state dimensions

    For processing inputs of system with single state dimension pass Tensor with shape: (B, T, 1) (unsqueezed tensor)
    """

    def __init__(
        self, n_time_steps: Optional[int] = None, norm: Optional[str] = None, dtype: torch.dtype = torch.cfloat
    ):
        """
        :param n_time_steps: number of time to produce steps, see torch.fft.fft2 for details
        :param norm: norm of FFT, see torch.fft.fft2 for details
        :param dtype: output data type, defaults to cfloat and should be complex
                      otherwise will cause loss of information after the RFFT transform
        """
        super(ComplexForwardFourier, self).__init__()

        self.n_time_steps = n_time_steps
        self.norm = norm
        self.dtype = dtype

        self.requires_grad_(False)  # non-trainable layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.fft.fft2(inputs, norm=self.norm, n=self.n_time_steps, dim=1)
        return outputs.to(self.dtype)


class RealBackwardFourier(torch.nn.Module):
    """
    Module computes Inverse Fourier Transform converting to real output
    It is used as a building block of FMLP and FLSTM networks

    It processes tensors with shape (B, F, S), where
        * B - batch
        * F - frequency modes
        * S - state dimensions

    For processing inputs of system with single state dimension pass Tensor with shape: (B, F, 1) (unsqueezed tensor)
    """

    def __init__(
        self, n_time_steps: Optional[int] = None, norm: Optional[str] = None, dtype: torch.dtype = torch.float32
    ):
        """
        :param n_time_steps: number of time to produce steps, see torch.fft.irfft for details
        :param norm: norm of IRFFT, see torch.fft.irfft for details
        :param dtype: output data type, for IRFFT should be real data type
        """
        super(RealBackwardFourier, self).__init__()

        self.n_time_steps = n_time_steps
        self.norm = norm
        self.dtype = dtype

        self.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.fft.irfft(inputs, n=self.n_time_steps, norm=self.norm, dim=1)
        return outputs.to(self.dtype)
