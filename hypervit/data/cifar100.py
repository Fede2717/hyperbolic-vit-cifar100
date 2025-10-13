from typing import Tuple, Any
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def get_transforms(train: bool = True, img_size: int = 32):
    """
    Build transforms for CIFAR-100.
    Train: RandomCrop + HorizontalFlip + ToTensor + Normalize
    Test:  ToTensor + Normalize
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                                 std=(0.2673, 0.2564, 0.2762)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                                 std=(0.2673, 0.2564, 0.2762)),
        ])

def get_cifar100_loaders(cfg: Any) -> Tuple[DataLoader, DataLoader]:
    train_set = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True,
        transform=get_transforms(True, cfg.img_size)
    )
    val_set = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True,
        transform=get_transforms(False, cfg.img_size)
    )

    pw = cfg.num_workers > 0  
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=(4 if pw else None),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader
