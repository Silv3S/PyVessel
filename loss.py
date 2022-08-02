import torch
import torch.nn as nn
import torch.nn.functional


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = torch.nn.functional.binary_cross_entropy(
            inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


class TverskyBCELoss(nn.Module):
    def __init__(self, alpha, beta, bce_ratio):
        """
        Tversky loss combined with Binary Cross Entropy. Parameters α and β
        control the magnitude of penalties for FPs and FNs, respectively.
        bce_ratio is responsible for weighting BCE and Tversky.
        """
        super(TverskyBCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_ratio = bce_ratio

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        tp = (inputs * targets).sum()
        fp = ((1-targets) * inputs).sum()
        fn = (targets * (1-inputs)).sum()
        tversky_loss = 1 - \
            ((tp + smooth) / (tp + self.alpha*fp + self.beta*fn + smooth))

        bce = torch.nn.functional.binary_cross_entropy(
            inputs, targets, reduction='mean')
        TverskyBCE = (self.bce_ratio * bce) + \
            ((1-self.bce_ratio) * tversky_loss)

        return TverskyBCE
