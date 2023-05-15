import torch
from torch.nn.modules.loss import _Loss


class PinballLoss(_Loss):
    """Class for the PinBall loss for quantile regression"""

    def __init__(self, loss_func, quantiles):
        """
        Args:
            loss_func : torch.nn._Loss
                Loss function to be used as the
                base loss for pinball loss
            quantiles : list
                list of quantiles estimated from the model
        """
        super().__init__()
        self.loss_func = loss_func
        self.quantiles = quantiles

    def forward(self, outputs, target):
        """
        Computes the pinball loss from forecasts
        Args:
            outputs : torch.tensor
                outputs from the model of dims (batch, no_quantiles, n_forecasts)
            target : torch.tensor
                actual targets of dims (batch, n_forecasts)

        Returns:
            float
                pinball loss
        """
        target = target.repeat(1, 1, len(self.quantiles))  # increase the quantile dimension of the targets
        differences = target - outputs
        base_losses = self.loss_func(outputs, target)  # dimensions - [n_batch, n_forecasts, no. of quantiles]
        positive_losses = (
            torch.tensor(self.quantiles, device=target.device).unsqueeze(dim=0).unsqueeze(dim=0) * base_losses
        )
        negative_losses = (
            1 - torch.tensor(self.quantiles, device=target.device).unsqueeze(dim=0).unsqueeze(dim=0)
        ) * base_losses
        pinball_losses = torch.where(differences >= 0, positive_losses, negative_losses)
        multiplier = torch.ones(size=(1, 1, len(self.quantiles)), device=target.device)
        multiplier[:, :, 0] = 2
        pinball_losses = multiplier * pinball_losses  # double the loss for the median quantile
        return pinball_losses
