import cv2
import torch

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader


class DroneDataset(Dataset):
    def __init__(self, img_dir, mask_dir, X, mean, std, transform=None, device='cpu'):
        super(DroneDataset, self).__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.X = X
        self.mean = mean
        self.std = std
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img = cv2.imread(self.img_dir + self.X[index][0] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_dir + self.X[index][0] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = Image.fromarray(augmented['image'])
            mask = augmented['mask']
        else:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()

        return img.to(self.device), mask.to(self.device)


def get_loader(img_dir, mask_dir, X, mean, std, transform=None, batch_size=16, shuffle=True, device='cpu'):
    dataset = DroneDataset(img_dir, mask_dir, X, mean, std, transform=transform, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
