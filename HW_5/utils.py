import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

def show_images(images, save_path, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]
    
    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def show_single_augmentation(original_img, augmented_img, save_path, title="Аугментация"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)

    if isinstance(original_img, torch.Tensor):
        original_img = transforms.ToPILImage()(original_img)
    orig_resized = resize_transform(original_img)
    orig_np = np.array(orig_resized) / 255.0

    if isinstance(augmented_img, torch.Tensor):
        augmented_img = transforms.ToPILImage()(augmented_img)
    aug_resized = resize_transform(augmented_img)
    aug_np = np.array(aug_resized) / 255.0
    
    # Оригинальное изображение
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')
    
    # Аугментированное изображение
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def show_multiple_augmentations(original_img, augmented_imgs, save_path, titles):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    
    # Оригинальное изображение
    orig_np = np.array(orig_resized)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')
    
    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        if isinstance(aug_img, torch.Tensor):
            aug_img = transforms.ToPILImage()(aug_img)
        aug_resized = resize_transform(aug_img)
        aug_np = np.array(aug_resized)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()