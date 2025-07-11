# Кастомные аугментации
import os

from torchvision import transforms
import random

from HW_5.datasets import CustomImageDataset
from HW_5.utils import show_single_augmentation

PLOTS_PATH = "results/ex_2"
os.makedirs(PLOTS_PATH, exist_ok=True)


class RandomGaussianBlur:
    """Случайное размытие Гауссом."""

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < 0.5:  # вероятность применения
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return transforms.functional.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img


class RandomPerspectiveTransform:
    """Случайная перспектива."""

    def __init__(self, distortion_scale=0.5, p=0.5):
        self.transform = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p)

    def __call__(self, img):
        return self.transform(img)


class RandomBrightnessContrast:
    """Случайная яркость и контрастность."""

    def __init__(self, brightness=0.5, contrast=0.5):
        self.transform = transforms.ColorJitter(brightness=brightness, contrast=contrast)

    def __call__(self, img):
        return self.transform(img)


train_dataset = CustomImageDataset('data/train', transform=None, target_size=(224, 224))

# Выбираем случайное изображение
idx = random.randint(0, len(train_dataset) - 1)
img, label = train_dataset[idx]

blur_aug = RandomGaussianBlur(kernel_size=5)
perspective_aug = RandomPerspectiveTransform(distortion_scale=0.5)
brightness_contrast_aug = RandomBrightnessContrast(brightness=0.5, contrast=0.5)

img_blur = blur_aug(img)
img_perspective = perspective_aug(img)
img_bc = brightness_contrast_aug(img)

aug_imgs = [img_blur, img_perspective, img_bc]
titles = ['Гауссово размытие', 'Перспектива', 'Яркость и контраст']

for j in range(len(titles)):
    show_single_augmentation(img, aug_imgs[j], f'{PLOTS_PATH}/{titles[j]}.png', titles[j])
