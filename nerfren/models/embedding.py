from abc import ABC, abstractmethod
import torch
from options import Configurable


class BaseEmbedding(ABC):
    @property
    @abstractmethod
    def out_channels(self):
        return 0


class PositionalEncoding(BaseEmbedding, Configurable):
    @staticmethod
    def modify_commandline_options(parser):
        return parser
    
    @property
    def out_channels(self):
        dim_out = self.in_channels * (len(self.funcs) * self.N_freqs)
        if not self.no_xyz:
            dim_out += self.in_channels
        return dim_out

    def __init__(self, in_channels, N_freqs, no_xyz=False, no_logscale=False):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.no_xyz = no_xyz
        self.no_logscale = no_logscale
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]

        if not self.no_logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def __call__(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = []
        if not self.no_xyz:
            out += [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)
