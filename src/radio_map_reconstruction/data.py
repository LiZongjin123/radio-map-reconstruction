from json import load
from pathlib import Path
from torch import Tensor, cat, float32, randperm, zeros_like
from torch.utils.data import Dataset
from os.path import join
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.v2.functional import to_dtype

ROOT = Path(__file__).resolve().parents[2]

class RadioDataset(Dataset):

    def __init__(self, part: str):
        super().__init__()
        path = join(ROOT, "run", f"{part}.json")
        with open(path, encoding="utf-8") as file:
            self.samples = load(file)

    def __len__(self):
        return len(self.samples) 

    def __getitem__(self, index: int) -> set[Tensor, dict[str, Tensor]]:
        gain_path, tx_path, building_map_path = self.samples[index] 

        gain = self.__read_image(gain_path)
        tx = self.__read_image(tx_path)
        building_map = self.__read_image(building_map_path)
        gain_mask = self.__random_sample(gain, tx, building_map, 0.1)

        label = {
            "gain": gain,
            "mask": (tx < 0.5) & (building_map < 0.5)
        }

        return cat((gain * gain_mask, tx, building_map, gain_mask)), label

    def __random_sample(
            self,
            gain: Tensor,
            tx: Tensor,
            building_map: Tensor,
            sample_rate: float
    ) -> Tensor:
        valid_area = (tx < 0.5) & (building_map < 0.5)
        valid_indices = valid_area.flatten().nonzero().squeeze(1)
        height, width = gain.shape[-2:]
        sample_count = round(height * width * sample_rate) 

        if valid_indices.numel() < sample_count:
            raise RuntimeError("有效采样点不够")

        selected = valid_indices[
            randperm(valid_indices.numel())[:sample_count]
        ]

        gain_mask = zeros_like(gain, dtype=float32)
        gain_mask.flatten()[selected] = 1

        return gain_mask

    def __read_image(self, path: str) -> Tensor:
        image = decode_image(
            path, 
            mode=ImageReadMode.GRAY
        )
        return to_dtype(
            image,
            dtype=float32,
            scale=True
        )

