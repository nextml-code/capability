import torch


def complex_conjugate(x):
    a = x[...,0]
    b = x[...,1]
    return torch.stack([a, -b], -1)


def complex_multiply(x1, x2):
    a, b = x1[...,0], x1[...,1]
    c, d = x2[...,0], x2[...,1]
    return torch.stack([
        a * c - b * d,
        a * d + b * c
    ], -1)


def correlation(x1, x2):
    signal_shape = x1.shape[-2:]
    x1 = torch.rfft(x1, 2)
    x2 = torch.rfft(x2, 2)
    corr = complex_multiply(x1, complex_conjugate(x2))
    return torch.irfft(corr, 2, signal_sizes=signal_shape)
