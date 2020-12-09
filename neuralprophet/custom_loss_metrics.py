import torch


class PinballLoss(torch.nn.Module):
    def __init__(self, quantiles=None):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, outputs, target):
        losses = []
        consecutive_diffs = []
        margin = 0.1
        for i, quantile in enumerate(self.quantiles):
            error = target - outputs[:, i, :]
            loss = torch.mean(torch.max((quantile - 1) * error, quantile * error), dim=-1)
            losses.append(loss)
            if i != 0:
                diff = torch.mean((outputs[:, i, :] - outputs[:, i - 1, :]), dim=-1)
                consecutive_diffs.append(diff)

        all_quantiles_loss = torch.sum(torch.stack(losses), dim=0)
        all_quantiles_penalty = torch.sum(
            torch.max(torch.tensor(0.0), 10 * (margin - torch.stack(consecutive_diffs))), dim=0
        )
        combined_loss = torch.mean(all_quantiles_loss + all_quantiles_penalty)
        return combined_loss
