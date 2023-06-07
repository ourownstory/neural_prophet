# from https://github.com/ts-kim/RevIN/blob/master/RevIN.py
import torch
import torch.nn as nn


class ReversibleNormalization(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, mode="instance"):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(ReversibleNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.mode = mode
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        if self.mode == "instance":
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        elif self.mode == "batch":
            self.mean = torch.mean(x, dim=[0, 1], keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=[0, 1], keepdim=True, unbiased=False) + self.eps).detach()
        else:
            raise ValueError("Reversible normalization allowed modes are 'instance' and 'batch'")

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
