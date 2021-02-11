import torch
from torch.nn.modules.loss import _Loss


class PinballLoss(_Loss):
    """Class for the PinBall loss for quantile regression"""

    def __init__(self, loss_func, quantiles=None):
        """
        Args:
            loss_func (torch.nn._Loss): Loss function to be used as the
                base loss for pinball loss
            quantiles (list): list of quantiles estimated from the model
        """
        super(PinballLoss, self).__init__()
        self.loss_func = loss_func
        self.quantiles = quantiles

    def forward(self, outputs, target):
        """
        Computes the pinball loss from forecasts
        Args:
            outputs (torch.tensor): outputs from the model of dims (batch, n_quantiles, n_forecasts)
            target (torch.tensor): actual targets of dims (batch, n_forecasts)

        Returns:
            pinball loss (float)
        """
        losses = []
        for i, quantile in enumerate(self.quantiles):
            diff = target - outputs[:, i, :]
            base_loss = self.loss_func(outputs[:, i, :], target)
            positive_loss = quantile * base_loss
            negative_loss = (1 - quantile) * base_loss
            pinball_loss = torch.mean(torch.where(diff >= 0, positive_loss, negative_loss), dim=-1)
            losses.append(pinball_loss)

        combined_loss = torch.mean(torch.sum(torch.stack(losses, dim=1), dim=-1))
        return combined_loss
