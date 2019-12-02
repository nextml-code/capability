import torch

from motion.architecture.complex_conjugate import complex_conjugate
from motion.architecture.complex_multiply import complex_multiply

def correlation(x1, x2):
    signal_shape = x1.shape[-2:]
    x1 = torch.rfft(x1, 2)
    x2 = torch.rfft(x2, 2)
    corr = complex_multiply(x1, complex_conjugate(x2))
    return torch.irfft(corr, 2, signal_sizes=signal_shape)
