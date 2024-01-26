from torch import nn
from torch import Tensor
from helper_functions import dice_loss


class DiceLoss(nn.Module):
    def __init__(self, multiclass: bool = True):
        super().__init__()
        self.multiclass = multiclass

    def forward(self, input: Tensor, target: Tensor):
        return dice_loss(input, target, self.multiclass)

# True masks shape: torch.Size([1, 640, 959])
# Predicted masks shape: torch.Size([1, 2, 640, 959])
