import torch


def saturating_sigmoid(x):
    return torch.clamp(1.2 * torch.sigmoid(x) - 0.1, 0, 1)


class KaiserStep(torch.nn.Module):
    def __init__(self, replacement=0.5, noise_scale=1):
        super().__init__()
        self.register_forward_hook(self.replace_forward)
        self.N = torch.distributions.Normal(0, noise_scale)
        self.bernoulli = torch.distributions.Bernoulli(replacement)

    def replace_forward(self, module, input, output):
        if self.training:
            replace = (self.bernoulli.sample(input[0].shape) == 1)
            output.data[replace] = (input[0][replace] > 0).float()
        else:
            output.data = (input[0] > 0).float()
        return output

    def forward(self, x):
        return saturating_sigmoid(x + self.N.sample(x.shape))
