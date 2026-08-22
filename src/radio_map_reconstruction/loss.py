from torch import Tensor
from torch.nn import Module

class RadioMapLoss(Module):

    def forward(self, outputs: Tensor, targets: dict[str, Tensor]) -> Tensor:
        gain = targets["gain"]
        mask = targets["mask"]
        if outputs.shape != gain.shape:
            raise ValueError("输出与标签的形状不相同")
        if outputs.shape != mask.shape:
            raise ValueError("输出与掩码形状不相同")

        squared_error = (outputs - gain).square()
        return squared_error.masked_select(mask).mean()