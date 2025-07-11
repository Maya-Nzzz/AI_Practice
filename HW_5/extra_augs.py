import os

import torch
import random
import numpy as np
import cv2
from PIL import Image, ImageOps

from HW_5.datasets import CustomImageDataset
from HW_5.utils import show_single_augmentation


class AddGaussianNoise:
    """Добавляет гауссов шум к изображению."""
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

class RandomErasingCustom:
    """Случайно затирает прямоугольную область изображения."""
    def __init__(self, p=0.5, scale=(0.02, 0.2)):
        self.p = p
        self.scale = scale
    def __call__(self, img):
        if random.random() > self.p:
            return img
        c, h, w = img.shape
        area = h * w
        erase_area = random.uniform(*self.scale) * area
        erase_w = int(np.sqrt(erase_area))
        erase_h = int(erase_area // erase_w)
        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)
        img[:, y:y+erase_h, x:x+erase_w] = 0
        return img

class CutOut:
    """Вырезает случайную прямоугольную область из изображения."""
    def __init__(self, p=0.5, size=(16, 16)):
        self.p = p
        self.size = size
    def __call__(self, img):
        if random.random() > self.p:
            return img
        c, h, w = img.shape
        cut_h, cut_w = self.size
        x = random.randint(0, w - cut_w)
        y = random.randint(0, h - cut_h)
        img[:, y:y+cut_h, x:x+cut_w] = 0
        return img

class Solarize:
    """Инвертирует пиксели выше порога."""
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, img):
        img_np = img.numpy()
        mask = img_np > self.threshold / 255.0
        img_np[mask] = 1.0 - img_np[mask]
        return torch.from_numpy(img_np)

class Posterize:
    """Уменьшает количество бит на канал."""
    def __init__(self, bits=4):
        self.bits = bits
    def __call__(self, img):
        img_np = np.array(img)
        factor = 2 ** (8 - self.bits)
        img_np = (img_np * 255).astype(np.uint8)
        img_np = (img_np // factor) * factor
        return torch.from_numpy(img_np.astype(np.float32).transpose(2, 0, 1) / 255.0)

class AutoContrast:
    """Автоматически улучшает контраст изображения."""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = img.numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil = ImageOps.autocontrast(img_pil)
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        return torch.from_numpy(img_np.transpose(2, 0, 1))

class ElasticTransform:
    """Эластичная деформация изображения."""
    def __init__(self, p=0.5, alpha=1, sigma=50):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma
    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img).transpose(1, 2, 0)
        h, w = img_np.shape[:2]
        
        # Создаем случайные смещения
        dx = np.random.randn(h, w) * self.alpha
        dy = np.random.randn(h, w) * self.alpha
        
        # Сглаживаем смещения
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)
        
        # Применяем деформацию
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x + dx
        y = y + dy
        
        # Нормализуем координаты
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        # Применяем трансформацию
        img_deformed = cv2.remap(img_np, x.astype(np.float32), y.astype(np.float32), 
                                cv2.INTER_LINEAR)
        return torch.from_numpy(img_deformed.transpose(1, 2, 0))

class MixUp:
    """Смешивает два изображения."""
    def __init__(self, p=0.5, alpha=0.2):
        self.p = p
        self.alpha = alpha

    def __call__(self, img1, img2):
        if random.random() > self.p:
            return img1

        lam = np.random.beta(self.alpha, self.alpha)

        # Преобразуем PIL Image в NumPy
        img1_np = np.array(img1).astype(np.float32)
        img2_np = np.array(img2).astype(np.float32)

        # Проверим размерность
        if img1_np.shape != img2_np.shape:
            raise ValueError("Размеры изображений должны совпадать для MixUp.")

        # Смешиваем
        mixed_np = lam * img1_np + (1 - lam) * img2_np
        mixed_np = mixed_np.astype(np.uint8)

        # Преобразуем обратно в PIL Image
        mixed_img = Image.fromarray(mixed_np)
        return mixed_img

if __name__ == '__main__':
    PLOTS_PATH = "results/ex_2"
    os.makedirs(PLOTS_PATH, exist_ok=True)

    train_dataset = CustomImageDataset('data/train', transform=None, target_size=(224, 224))

    idx = random.randint(0, len(train_dataset) - 1)
    idx1 = random.randint(0, len(train_dataset) - 1)
    img, label = train_dataset[idx]
    img1, label1 = train_dataset[idx1]

    posterize_aug = Posterize()
    elasticTransform_aug = ElasticTransform()
    mixup_aug = MixUp()

    posterize_img = posterize_aug(img)
    elasticTransform_img = elasticTransform_aug(img)
    mixup_img = mixup_aug(img, img1)

    aug_imgs = [posterize_img, elasticTransform_img, mixup_img]
    titles = ['Уменьшено количество бит на канал', 'Эластичная деформация', 'Смешивание']

    for j in range(len(titles)):
        show_single_augmentation(img, aug_imgs[j], f'{PLOTS_PATH}/extra_augs_{titles[j]}.png', titles[j])