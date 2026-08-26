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
        error_per_sample = (squared_error * mask).flatten(start_dim=1).sum(dim=1)
        valid_pixels_per_sample = mask.flatten(start_dim=1).sum(dim=1)
        return (error_per_sample / valid_pixels_per_sample).mean()
