import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Optional


def plot_training_history(
        history: Dict[str, List[float]],
        save_path: Optional[str],
        title: Optional[str]
):
    """
    Визуализирует историю обучения с двумя подграфиками: Loss и Accuracy.

    history должен содержать:
        'train_losses', 'test_losses', 'train_accs', 'test_accs'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy plot
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def plot_heatmap(
        data,
        x_labels: List[str],
        y_labels: List[str],
        save_path: Optional[str],
        title: Optional[str]
):
    """
    Строит тепловую карту (например, confusion matrix).
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels, annot=True, fmt='.2f', cmap='viridis')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
