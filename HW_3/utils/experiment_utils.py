import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import os
from typing import Dict, Any

# Настройка логирования: вывод в консоль с указанием времени, уровня и сообщения
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def setup_experiment_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    Создает и настраивает логгер для эксперимента.

    Args:
        log_dir (str): Путь к директории для логов.
        experiment_name (str): Имя эксперимента (используется как имя файла лога).

    Returns:
        logging.Logger: Настроенный объект логгера.
    """
    os.makedirs(log_dir, exist_ok=True)  # Создать директорию, если не существует
    logger = logging.getLogger(experiment_name)
    fh = logging.FileHandler(os.path.join(log_dir, f'{experiment_name}.log'))  # Файл лога
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)  # Добавляем хендлер, если его еще нет
    return logger


def log_experiment_results(logger: logging.Logger, results: Dict[str, Any]):
    logger.info('Experiment results: %s', results)


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


def train_model(model, train_loader, test_loader, logger, epochs=10, lr=0.001, device='cpu'):
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
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

        # Логируем информацию по эпохе
        logger.info(f'Epoch {epoch + 1}/{epochs}')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        logger.info('-' * 50)

    # Записываем финальные результаты
    log_experiment_results(logger, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    })

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }
