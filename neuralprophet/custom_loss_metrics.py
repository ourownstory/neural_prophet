import torch


class PinballLoss(torch.nn.Module):
    def __init(self, quantiles=None, reduction='mean'):
        super(PinballLoss, self).__init__()
        if quantiles is None:
            self.quantiles = [0.50]
        else:
            self.quantiles = quantiles
        self.reduction = reduction

    def forward(self, outputs, target):
        losses = list()
        for i in range(len(self.quantiles)):
            output = outputs[i]
            tau = self.quantiles[i]
            error = target - output
            loss = torch.max((tau * error), (1-tau) * error)
            losses.append(loss)
    
        if self.reduction == 'sum':
            combined_loss = torch.sum(torch.tensor(losses))
        if self.reduction == 'mean':
            combined_loss = torch.mean(torch.tensor(losses))
        return combined_loss
