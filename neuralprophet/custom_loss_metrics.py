import torch


class PinballLoss(torch.nn.Module):
    """Class for the PinBall loss for quantile regression"""

    def __init__(self, quantiles=None):
        """
        Args:
            quantiles (list): list of quantiles estimated from the model
        """
        super(PinballLoss, self).__init__()
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
            error = target - outputs[:, i, :]
            loss = torch.mean(torch.max((quantile - 1) * error, quantile * error), dim=-1)
            losses.append(loss)

        combined_loss = torch.mean(torch.sum(torch.stack(losses, dim=1), dim=-1))
        return combined_loss
