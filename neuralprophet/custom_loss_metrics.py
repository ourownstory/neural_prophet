import torch


class PinballLoss(torch.nn.Module):
    def __init__(self, quantiles=None):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, outputs, target):
        losses = []
        for i, quantile in enumerate(self.quantiles):
            error = target - outputs[:, i, :]
            loss = torch.mean(torch.max((quantile - 1) * error, quantile * error), dim=-1, keepdim=True)
            losses.append(loss)

        # if self.reduction == "sum":
        #     combined_loss = torch.sum(torch.tensor(losses))
        # else:
        #     combined_loss = torch.mean(torch.stack(losses))
        losses = torch.cat(losses, dim=1)
        losses = torch.mean(torch.sum(losses, dim=-1))
        return losses
