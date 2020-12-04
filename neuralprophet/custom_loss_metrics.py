import torch


class PinballLoss(torch.nn.Module):
    def __init__(self, quantiles=None):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, outputs, target):
        losses = []
        for i, quantile in enumerate(self.quantiles):
            error = target - outputs[:, i, :]
            loss = torch.mean(torch.max((quantile - 1) * error, quantile * error))
            losses.append(loss)

        combined_loss = torch.mean(torch.stack(losses))
        return combined_loss
