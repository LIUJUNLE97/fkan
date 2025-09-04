import math
from typing import Optional

import torch
import torch.nn as nn
import torch.fft


# -----------------------------
# Fourier-Kolmogorov–Arnold Network (FKAN) for 1D time-series (B, T, C)
# Base: Spectral (Fourier) operator over time + KAN-style univariate basis nonlinearity
# Author: Junle Liu, Yanyu Ke, Haoyan Li, Wenliang Chen, Tianle Niu, K.T. Tse, Gang Hu
# -----------------------------


class SpectralConv1d(nn.Module):
    """
    1D spectral convolution over the time axis using learned complex weights for low modes.
    Only applied to low modes

    Input shape: (B, C_in, T)
    Output shape: (B, C_out, T)
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels # the pressure taps number  26 here 
        self.out_channels = out_channels
        self.modes = modes
        # Learnable complex weights for positive frequencies [0..modes-1]
        # We'll store as real tensors of shape (..., 2) representing (real, imag)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, 2) * 0.02
        )

    def compl_mul(self, inp_fft: torch.Tensor) -> torch.Tensor:
        # inp_fft: (B, C_in, K) complex
        B, Cin, K = inp_fft.shape
        Kuse = min(K, self.modes)
        inp_k = inp_fft[:, :, :Kuse]  # (B, Cin, Kuse)
        w = self.weight[:, :, :Kuse, :]  # (Cin, Cout, Kuse, 2)
        w_complex = torch.view_as_complex(w.contiguous())  # (Cin, Cout, Kuse)
        # Einstein: b i k , i o k -> b o k
        out_k = torch.einsum('bik,iok->bok', inp_k, w_complex)
        # Pad back to K with zeros on remaining high modes
        if Kuse < K:
            pad = torch.zeros(B, self.out_channels, K - Kuse, dtype=out_k.dtype, device=out_k.device)
            out_fft = torch.cat([out_k, pad], dim=-1)
        else:
            out_fft = out_k
        return out_fft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, T)
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, Cin, K) complex
        out_ft = self.compl_mul(x_ft)
        x_out = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)  # (B, Cout, T)
        return x_out


class KANActivation(nn.Module):
    """
    Kolmogorov–Arnold style univariate function approximator using a Fourier basis per channel.

    For each element x, we compute: 
        phi(x) = sum_{k=1..K} [a_k * sin(k*x) + b_k * cos(k*x)] + c1 * x + c2 * x^2 + b
    with parameters (a_k, b_k, c1, c2, b) learned independently for each channel.

    Works elementwise on tensors of shape (B, C, T). Returns same shape.
    """

    def __init__(self, channels: int, max_freq: int = 6):
        super().__init__()
        self.channels = channels
        self.max_freq = max_freq
        # Parameters per channel
        self.a = nn.Parameter(torch.zeros(channels, max_freq))
        self.b = nn.Parameter(torch.zeros(channels, max_freq))
        self.c1 = nn.Parameter(torch.zeros(channels))
        self.c2 = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        # Initialize small to start near identity
        nn.init.normal_(self.a, std=1e-3)
        nn.init.normal_(self.b, std=1e-3)
        nn.init.normal_(self.c1, std=1e-3)
        nn.init.normal_(self.c2, std=1e-3)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        assert C == self.channels, f"KANActivation expected {self.channels} channels, got {C}"
        # Expand for broadcasting: (B, C, T, 1)
        x_exp = x.unsqueeze(-1)
        ks = torch.arange(1, self.max_freq + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        sin_terms = torch.sin(ks * x_exp)  # (B, C, T, K)
        cos_terms = torch.cos(ks * x_exp)  # (B, C, T, K)
        # Weight and sum over frequency dimension
        out = (sin_terms * self.a.view(1, C, 1, -1)).sum(dim=-1) + (cos_terms * self.b.view(1, C, 1, -1)).sum(dim=-1)
        out = out + self.c1.view(1, C, 1) * x + self.c2.view(1, C, 1) * (x * x) + self.bias.view(1, C, 1)
        return out


class KANBlock(nn.Module):
    """Pointwise mixer using 1x1 conv -> KANActivation -> 1x1 conv with residual."""

    def __init__(self, channels: int, hidden: int, max_freq: int = 6, dropout: float = 0.0):
        super().__init__()
        self.pre = nn.Conv1d(channels, hidden, kernel_size=1)
        self.act = KANActivation(hidden, max_freq=max_freq)
        self.post = nn.Conv1d(hidden, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        y = self.act(y)
        y = self.post(y)
        y = self.drop(y)
        y = self.bn(y + x)
        return y


class FKANLayer(nn.Module):
    """One block: spectral conv along time + KAN mixer + residual skip."""

    def __init__(self, width: int, modes: int, kan_hidden: Optional[int] = None, max_freq: int = 6, dropout: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.kan = KANBlock(width, hidden=kan_hidden or (2 * width), max_freq=max_freq, dropout=dropout)
        self.skip = nn.Conv1d(width, width, kernel_size=1)
        self.bn = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.kan(x) + self.skip(x)
        return self.bn(torch.relu(y))


class FourierKAN(nn.Module):
    """
    End-to-end model for input (B, T, C_in) -> output (B, T, C_out), default C_in=C_out.

    This implementation treats the 26 spatial points as feature channels and applies
    Fourier (spectral) mixing along the 1024-length time dimension.
    """

    def __init__(
        self,
        in_channels: int = 26,
        out_channels: int = 26,
        width: int = 64,
        modes: int = 64,
        depth: int = 4,
        kan_hidden: Optional[int] = None,
        max_freq: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.layers = nn.ModuleList([
            FKANLayer(width, modes=modes, kan_hidden=kan_hidden, max_freq=max_freq, dropout=dropout)
            for _ in range(depth)
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C_in)
        returns: (B, T, C_out)
        """
        assert x.dim() == 3, f"Expected (B, T, C), got {x.shape}"
        # Move to (B, C, T)
        x = x.transpose(1, 2)
        y = self.lift(x)
        for layer in self.layers:
            y = layer(y)
        y = self.proj(y)
        # Back to (B, T, C)
        y = y.transpose(1, 2)
        return y

'''
# -----------------------------
# Minimal self-test
# -----------------------------
if __name__ == "__main__":
    B, T, C = 2, 1024, 26
    model = FourierKAN(in_channels=C, out_channels=C, width=64, modes=96, depth=4, max_freq=8, dropout=0.05)
    x = torch.randn(B, T, C)
    with torch.no_grad():
        y = model(x)
    print("Input:", x.shape, "Output:", y.shape)
'''