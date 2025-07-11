import math

import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Optional

import torch
from sklearn.metrics import confusion_matrix


def plot_training_history(
        history: Dict[str, List[float]],
        num_epochs: int,
        save_path: Optional[str],
        title: Optional[str]
):
    """
    Визуализирует историю обучения с двумя подграфиками: Loss и Accuracy.

    history должен содержать:
        'train_losses', 'test_losses', 'train_accs', 'test_accs'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = [x for x in range(1, num_epochs+1)]
    # Loss plot
    ax1.plot(epochs, history['train_losses'], marker='o', linestyle='-', label='Train Loss')
    ax1.plot(epochs, history['test_losses'], marker='o', linestyle='-', label='Test Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history['train_accs'], marker='o', linestyle='-', label='Train Acc')
    ax2.plot(epochs, history['test_accs'], marker='o', linestyle='-', label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(model, save_path, loader, device, classes, title):
    """
    Строит матрицу ошибок для модели на loader.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.numpy())
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(title)
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


def visualize_activations(neural_layer, input_data_batch, save_path, display_count=8, chart_title=None):
    """
    Отображает карты активаций, полученные после применения заданного нейронного слоя к входным данным.

    :param neural_layer: Экземпляр nn.Module (например, сверточный слой из модели)
    :param input_data_batch: Тензор входных данных в формате батча [N, C, H, W]
    :param display_count: Количество карт признаков для отображения
    :param chart_title: Заголовок для всего графика
    """
    # Переключение слоя в режим оценки (inference) и отключение расчета градиентов
    neural_layer.eval()
    with torch.no_grad():
        # Извлечение первого элемента из батча и добавление размерности батча,
        # затем перемещение на то же устройство, что и параметры слоя.
        single_input_item = input_data_batch[0].unsqueeze(0).to(next(neural_layer.parameters()).device)
        # Получение активаций после прохода через слой
        layer_activations = neural_layer(single_input_item)
        # Перемещение активаций на CPU и удаление размерности батча
        feature_maps_data = layer_activations.cpu().squeeze(0)

    # Настройка параметров сетки для отображения
    columns_for_grid = min(4, display_count)
    rows_for_grid = math.ceil(display_count / columns_for_grid)

    # Создание фигуры для графиков
    plt.figure(figsize=(columns_for_grid * 3.5, rows_for_grid * 3.5))  # Увеличил размер для лучшей читаемости

    # Проход по картам признаков и их отображение
    for i in range(display_count):
        # Проверка, если количество запрошенных карт превышает доступное
        if i >= feature_maps_data.shape[0]:
            break
        plt.subplot(rows_for_grid, columns_for_grid, i + 1)
        plt.imshow(feature_maps_data[i], cmap='plasma')  # Изменил цветовую схему
        plt.title(f'Карта {i + 1}')  # Добавил подзаголовок для каждой карты
        plt.axis('off')  # Отключение осей

    # Установка общего заголовка для графика, если он задан
    if chart_title:
        plt.suptitle(chart_title, fontsize=16)  # Увеличил размер шрифта заголовка

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Корректировка отступов, чтобы заголовок не наезжал

    plt.savefig(save_path)
    plt.close()
