from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Any, Any
from utils import normalize

class CIFAR10_Dataset(Dataset):
    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        img = self.to_tensor(img)
        img = normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        return img, target

    def __len__(self) -> int:
        return len(self.data)