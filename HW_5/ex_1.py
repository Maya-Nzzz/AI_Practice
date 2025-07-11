# Пайплайн стандартных аугментаций
import os
import random
from PIL import Image

import torchvision.transforms as transforms

from HW_5.utils import show_single_augmentation

# Путь к папке с изображениями
train_dir = 'data/train'
PLOTS_PATH = "results/ex_1"
os.makedirs(PLOTS_PATH, exist_ok=True)

# Получаем список классов
classes = os.listdir(train_dir)
classes = [cls for cls in classes if os.path.isdir(os.path.join(train_dir, cls))]

# Выбираем по одному изображению из 5 разных классов
selected_images = []
for cls in random.sample(classes, 5):
    cls_folder = os.path.join(train_dir, cls)
    images = [img for img in os.listdir(cls_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]
    if images:
        img_path = os.path.join(cls_folder, random.choice(images))
        selected_images.append(img_path)

# Отдельные аугментации
aug_flip = transforms.RandomHorizontalFlip(p=1.0)
aug_crop = transforms.RandomCrop(size=224)
aug_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
aug_rotation = transforms.RandomRotation(degrees=45)
aug_gray = transforms.RandomGrayscale(p=1.0)

# Все аугментации вместе
combined_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomCrop(size=224),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.RandomRotation(degrees=45),
    transforms.RandomGrayscale(p=1.0)
])


# Применяем к каждому выбранному изображению
for i in range(len(selected_images)):
    img = Image.open(selected_images[i]).convert("RGB")

    flip_img = aug_flip(img)
    crop_img = aug_crop(img)
    jitter_img = aug_jitter(img)
    rotate_img = aug_rotation(img)
    gray_img = aug_gray(img)
    combined_img = combined_transform(img)

    aug_imgs = [flip_img, crop_img, jitter_img, rotate_img, gray_img, combined_img]
    titles = ["Flip", "Crop", "Jitter", "Rotation", "Gray", "Combined"]

    for j in range(len(titles)):
        show_single_augmentation(img, aug_imgs[j], f'{PLOTS_PATH}/{i}_{titles[j]}.png', titles[j])