import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader


def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False):
    """
    Запускает одну эпоху обучения или теста.

    Args:
        model: Модель PyTorch.
        data_loader: DataLoader с данными.
        criterion: Функция потерь.
        optimizer: Оптимизатор (используется только при обучении).
        device (str): Устройство (cpu или cuda).
        is_test (bool): Если True — режим теста, иначе обучение.

    Returns:
        Tuple[float, float]: Среднее значение потерь и точность за эпоху.
    """
    # Переключаем модель в режим eval или train
    model.eval() if is_test else model.train()

    total_loss = 0
    correct = 0
    total = 0

    # Итерация по батчам
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()  # Обнуляем градиенты

        output = model(data)  # Прямой проход
        loss = criterion(output, target)  # Вычисляем потери

        if not is_test and optimizer is not None:
            loss.backward()  # Обратное распространение
            optimizer.step()  # Обновляем веса

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # Предсказанный класс
        correct += pred.eq(target.view_as(pred)).sum().item()  # Количество правильных
        total += target.size(0)  # Общее количество

    # Возвращаем среднюю потерю и точность
    return total_loss / len(data_loader), correct / total


def train_model(model, train_loader, test_loader, weight_decay=0.0, epochs=5, lr=0.001, device='cpu'):
    """
    Запускает цикл обучения модели с периодическим тестированием и логированием.

    Args:
        model: Модель PyTorch.
        train_loader: DataLoader с обучающими данными.
        test_loader: DataLoader с тестовыми данными.
        logger: Экземпляр логгера для записи информации.
        epochs (int): Количество эпох.
        lr (float): Скорость обучения.
        device (str): Устройство (cpu или cuda).

    Returns:
        Dict[str, list]: История потерь и точностей на train и test за все эпохи.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        # Обучающая эпоха
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        # Тестовая эпоха
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)

        # Сохраняем метрики
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }


def calculate_accuracy(data_loader: DataLoader, model_to_evaluate: torch.nn.Module,
                       device_to_use: torch.device):
    model_to_evaluate.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device_to_use), labels.to(device_to_use)
            outputs = model_to_evaluate(inputs)
            _, predicted_labels = torch.max(outputs, 1) # Получить индекс максимальной логарифмической вероятности
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    return correct_predictions / total_samples
