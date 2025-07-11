import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os
import gc

from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """Кастомный датасет для работы с папками классов"""

    def __init__(self, root_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            root_dir (str): Путь к папке с классами
            transform: Аугментации для изображений
            target_size (tuple): Размер для ресайза изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        # Получаем список классов (папок)
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Собираем все пути к изображениям
        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')

        # Ресайзим изображение
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        # Применяем аугментации
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        """Возвращает список имен классов"""
        return self.classes


    # Функция для оценки памяти процесса
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024



sizes = [64, 128, 224, 512]
times = []
memories = []
train_dataset = CustomImageDataset('data/train', transform=None, target_size=(224, 224))
PLOTS_PATH = "results/ex_5"
os.makedirs(PLOTS_PATH, exist_ok=True)

# Аугментации
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

for sz in sizes:
    # Сброс сборщика мусора
    gc.collect()
    torch.cuda.empty_cache()

    # Создаем 100 изображений
    images = [train_dataset[i][0] for i in range(100)]

    start_mem = get_memory_usage_mb()
    start_time = time.time()

    augmented_images = []
    for img in images:
        aug_img = augmentation(img)
        augmented_images.append(aug_img)

    end_time = time.time()
    end_mem = get_memory_usage_mb()

    elapsed_time = end_time - start_time
    mem_diff = end_mem - start_mem

    times.append(elapsed_time)
    memories.append(mem_diff if mem_diff > 0 else 0.1)  # чтобы не было отрицательных значений

    print(f"Size {sz}x{sz}: Time = {elapsed_time:.2f} sec, Memory ≈ {mem_diff:.2f} MB")

# График времени
plt.figure(figsize=(8, 5))
plt.plot(sizes, times, marker='o')
plt.title('Зависимость времени от размера изображений')
plt.xlabel('Размер изображения (px)')
plt.ylabel('Время обработки (сек)')
plt.grid(True)
plt.savefig(f'{PLOTS_PATH}/time.png')
plt.close()

# График памяти
plt.figure(figsize=(8, 5))
plt.plot(sizes, memories, marker='o', color='orange')
plt.title('Зависимость памяти от размера изображений')
plt.xlabel('Размер изображения (px)')
plt.ylabel('Дополнительная память (MB)')
plt.grid(True)
plt.savefig(f'{PLOTS_PATH}/memory.png')
plt.close()
