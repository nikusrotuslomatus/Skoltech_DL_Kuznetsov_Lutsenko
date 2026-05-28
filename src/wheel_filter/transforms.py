from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageOps


class SquarePad:
    def __init__(self, fill: int | tuple[int, int, int] = 0) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


def build_transforms(cfg: dict, train: bool):
    try:
        from torchvision import transforms as T
    except Exception as exc:
        raise RuntimeError("torchvision is required for deep training transforms") from exc

    p = cfg["preprocessing"]
    size = int(p["input_size"])
    fill = int(p.get("pad_fill", 0))
    mean: Sequence[float] = p["imagenet_mean"]
    std: Sequence[float] = p["imagenet_std"]

    if train:
        return T.Compose(
            [
                SquarePad(fill=fill),
                T.Resize((size + 32, size + 32)),
                T.RandomResizedCrop(size=size, scale=(0.85, 1.0), ratio=(0.85, 1.18)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.08),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    return T.Compose(
        [
            SquarePad(fill=fill),
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

