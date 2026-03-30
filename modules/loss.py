import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,pred,y):
        pred = torch.softmax(pred,dim=1)
        target = F.one_hot(y,2).permute(0, 3, 1, 2).float()
        up = (pred*target).sum(dim=(2,3))
        down = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = 2 * up / down
        return 1 - dice.mean()

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = DiceLoss()
        self.c2 = nn.CrossEntropyLoss()
    def forward(self,pred,y):
        return (self.c1(pred,y) + self.c2(pred,y)) * 0.5
