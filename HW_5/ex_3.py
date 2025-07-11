# Анализ датасета
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


data_dir = "data/train"
class_counts = {}
widths = []
heights = []

# Проходим по каждой папке-классу
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    class_counts[class_name] = len(image_files)

    for img_name in image_files:
        img_path = os.path.join(class_path, img_name)
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception as e:
            print(f"Ошибка при открытии {img_path}: {e}")

# Статистика по размерам
min_width, max_width, mean_width = np.min(widths), np.max(widths), np.mean(widths)
min_height, max_height, mean_height = np.min(heights), np.max(heights), np.mean(heights)

print(f"Минимальная ширина: {min_width}, максимальная: {max_width}, средняя: {mean_width:.2f}")
print(f"Минимальная высота: {min_height}, максимальная: {max_height}, средняя: {mean_height:.2f}")

plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xticks(rotation=45)
plt.title("Количество изображений по классам")
plt.xlabel("Класс")
plt.ylabel("Количество")
plt.tight_layout()
plt.savefig("results/ex_3/Количество изображений.png")
plt.close()

# Распределение ширин и высот
plt.figure(figsize=(10, 5))
plt.hist(widths, bins=30, alpha=0.5, label='Ширины')
plt.hist(heights, bins=30, alpha=0.5, label='Высоты')
plt.title("Распределение размеров изображений")
plt.xlabel("Пиксели")
plt.ylabel("Количество")
plt.legend()
plt.savefig("results/ex_3/Размеры.png")
plt.close()
