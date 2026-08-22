from os import listdir
from torch import Tensor, cat, float32, randperm, zeros_like
from torch.utils.data import Dataset
from os.path import join
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.v2.functional import to_dtype

class RadioDataset(Dataset):

    def __init__(self, root: str, partition: str):
        super().__init__()
        self.gain_dir = join(root, "gain", partition)
        self.tx_dir = join(root, "png", "antennas")
        self.building_map_dir = join(root, "png", "buildings_complete")

        gain_names = set(listdir(self.gain_dir))
        tx_names = set(listdir(self.tx_dir))
        building_map_names = set(listdir(self.building_map_dir))

        if len(gain_names) != len(tx_names):
            raise RuntimeError("Reference Radio Map的数量和Tx Map不一致。")
        
        self.samples = []
        for gain_name in gain_names:
            if gain_name not in tx_names:
                raise RuntimeError("存在不匹配的Reference Radio Map和Tx Map。")
            building_map_index = gain_name.split("_")[0]
            building_map_name = f"{building_map_index}.png"
            if building_map_name not in building_map_names:
                raise RuntimeError("存在不匹配的Reference Radio Map和Building Map")
            self.samples.append((gain_name, gain_name, building_map_name))

    def __len__(self):
        return len(self.samples) 

    def __getitem__(self, index: int) -> set[Tensor, dict[str, Tensor]]:
        gain_name, tx_name, building_map_name = self.samples[index] 

        gain = self.__read_image(join(self.gain_dir, gain_name))
        tx = self.__read_image(join(self.tx_dir, tx_name))
        building_map = self.__read_image(join(self.building_map_dir, building_map_name))
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

