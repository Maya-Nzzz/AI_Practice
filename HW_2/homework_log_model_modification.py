import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


class LogisticRegression(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        """
        Инициализирует модель логистической регрессии.

        Args:
            in_features: Количество входных признаков.
            num_classes: Количество выходных классов. Для бинарной классификации
                         это 1, если используется BCEWithLogitsLoss, или 2, если
                         используется CrossEntropyLoss. Для многоклассовой -
                         фактическое количество классов.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через модель.

        Args:
            x: Входной тензор формы (batch_size, in_features).

        Returns:
            Тензор логитов формы (batch_size, num_classes).
        """
        return self.linear(x)


class ClassificationDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        Инициализирует набор данных для классификации.

        Args:
            X: Тензор признаков.
            y: Тензор целевых меток (классов).
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Возвращает общее количество образцов в наборе данных.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Извлекает образец из набора данных по заданному индексу.

        Args:
            idx: Индекс образца для извлечения.

        Returns:
            Кортеж, содержащий тензор признаков и тензор целевой метки.
        """
        return self.X[idx], self.y[idx]


def make_classification_data(n_samples: int = 100, n_features: int = 2,
                             n_classes: int = 2, source: str = 'random') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Генерирует синтетические данные для классификации или загружает реальный набор данных.

    Args:
        n_samples: Количество образцов для генерации ('random' source).
        n_features: Количество признаков для генерации ('random' source).
        n_classes: Количество классов для генерации ('random' source).
        source: Тип данных для генерации ('random', 'breast_cancer').

    Returns:
        Кортеж (признаки, метки) в виде torch.Tensor.
    """
    if source == 'random':
        # make_classification возвращает X как float64, y как int64
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features, # Все признаки информативны
            n_redundant=0,
            n_repeated=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42
        )
        X = torch.tensor(X, dtype=torch.float32)
        # CrossEntropyLoss ожидает LongTensor для меток
        y = torch.tensor(y, dtype=torch.long)
        return X, y
    elif source == 'breast_cancer':
        data = load_breast_cancer()
        X = torch.tensor(data['data'], dtype=torch.float32)
        # Для breast_cancer y - это 0 или 1. CrossEntropyLoss ожидает 1D тензор для меток.
        # Если оригинальный y был (N,1), его нужно сжать до (N,).
        y = torch.tensor(data['target'], dtype=torch.long)
        return X, y
    else:
        raise ValueError(f"Неизвестный источник данных: {source}. Выберите 'random' или 'breast_cancer'.")


def calculate_metrics(y_true: np.ndarray, y_pred_classes: np.ndarray, y_pred_probs: np.ndarray,
                      num_classes: int) -> dict:
    """
    Вычисляет различные метрики классификации.

    Args:
        y_true: Истинные метки (numpy array).
        y_pred_classes: Предсказанные метки классов (numpy array).
        y_pred_probs: Предсказанные вероятности для каждого класса (numpy array).
        num_classes: Общее количество классов.

    Returns:
        Словарь, содержащий вычисленные метрики.
    """
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred_classes)

    # Precision, Recall, F1-score
    # Для многоклассовой классификации используем 'macro' или 'weighted' среднее.
    # 'weighted' учитывает дисбаланс классов.
    average_type = 'binary' if num_classes == 2 else 'weighted'

    metrics['precision'] = precision_score(y_true, y_pred_classes, average=average_type, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_classes, average=average_type, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred_classes, average=average_type, zero_division=0)

    if num_classes == 2:
        # Для бинарной классификации y_pred_probs должен быть вероятностями положительного класса (класса 1)
        # Если y_pred_probs имеет форму (N, 2), берем вероятность класса 1.
        if y_pred_probs.ndim == 2:
            y_score_for_auc = y_pred_probs[:, 1]
        else: # Должен быть (N,) для бинарной
            y_score_for_auc = y_pred_probs
        metrics['roc_auc'] = roc_auc_score(y_true, y_score_for_auc)
    else:
        # Многоклассовый ROC-AUC (стратегия one-vs-rest со средним 'weighted')
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='weighted')

    return metrics

def log_epoch(epoch: int, loss: float, **metrics: float) -> None:
    """
    Логирует информацию об эпохе, включая потери и другие метрики.

    Args:
        epoch: Номер текущей эпохи.
        loss: Средние потери для эпохи.
        **metrics: Дополнительные метрики для логирования (например, accuracy, precision).
    """
    msg = f"Epoch {epoch:03d}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)

def plot_confusion_matrix(y_true: np.ndarray, y_pred_classes: np.ndarray, class_names: list[str] = None,
                          title: str = "Confusion Matrix") -> None:
    """
    Строит матрицу ошибок.

    Args:
        y_true: Истинные метки.
        y_pred_classes: Предсказанные метки.
        class_names: Список имен классов для меток на графике.
        title: Заголовок графика.
    """
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()


def train_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, epochs: int, num_classes: int) -> None:
    """
    Обучает модель классификации.

    Args:
        model: Обучаемая модель PyTorch.
        dataloader: DataLoader для обучающих данных.
        criterion: Функция потерь.
        optimizer: Оптимизатор.
        epochs: Количество эпох обучения.
        num_classes: Количество выходных классов для модели.
    """
    model.train() # Устанавливаем модель в режим обучения
    print("\n--- Начало обучения модели ---")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        all_y_true = []
        all_y_pred_classes = []
        all_y_pred_probs = []

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_X)
            # CrossEntropyLoss ожидает логиты (N, C) и метки (N)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Для расчета метрик
            probs = torch.softmax(logits, dim=1) # Получаем вероятности
            predicted_classes = torch.argmax(probs, dim=1) # Получаем индекс предсказанного класса

            all_y_true.extend(batch_y.cpu().numpy())
            all_y_pred_classes.extend(predicted_classes.cpu().numpy())
            all_y_pred_probs.extend(probs.detach().cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        metrics = calculate_metrics(
            np.array(all_y_true),
            np.array(all_y_pred_classes),
            np.array(all_y_pred_probs),
            num_classes
        )

        if epoch % 10 == 0 or epoch == epochs:
            log_epoch(epoch, avg_loss, **metrics)
    print("--- Обучение завершено ---")

def evaluate_model(model: nn.Module, dataloader: DataLoader, num_classes: int,
                   class_names: list[str] = None) -> dict:
    """
    Оценивает модель на заданном наборе данных и строит матрицу ошибок.

    Args:
        model: Обученная модель PyTorch.
        dataloader: DataLoader для данных оценки.
        num_classes: Количество выходных классов для модели.
        class_names: Необязательный список имен классов для матрицы ошибок.

    Returns:
        Словарь, содержащий метрики оценки.
    """
    model.eval() # Устанавливаем модель в режим оценки
    print("\n--- Начало оценки модели ---")
    all_y_true = []
    all_y_pred_classes = []
    all_y_pred_probs = []

    with torch.no_grad(): # Отключаем вычисление градиентов во время оценки
        for batch_X, batch_y in dataloader:
            logits = model(batch_X)
            probs = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probs, dim=1)

            all_y_true.extend(batch_y.cpu().numpy())
            all_y_pred_classes.extend(predicted_classes.cpu().numpy())
            all_y_pred_probs.extend(probs.cpu().numpy())

    metrics = calculate_metrics(
        np.array(all_y_true),
        np.array(all_y_pred_classes),
        np.array(all_y_pred_probs),
        num_classes
    )

    print("\n--- Результаты оценки ---")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').capitalize()}: {v:.4f}")

    plot_confusion_matrix(
        np.array(all_y_true),
        np.array(all_y_pred_classes),
        class_names=class_names,
        title="Матрица ошибок (набор оценки)"
    )
    print("--- Оценка завершена ---")
    return metrics


def main():
    N_SAMPLES = 500
    N_FEATURES = 4
    N_CLASSES = 3
    DATA_SOURCE = 'breast_cancer'
    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 0.1
    MODEL_PATH = 'models/logreg_torch_refactored.pth'

    # Динамическое определение num_classes для breast_cancer
    if DATA_SOURCE == 'breast_cancer':
        data = load_breast_cancer()
        N_FEATURES = data['data'].shape[1]
        N_CLASSES = len(np.unique(data['target']))
        class_names = data['target_names'].tolist()
    else: # Для 'random' данных
        class_names = [f'Class {i}' for i in range(N_CLASSES)]

    print(f"--- Конфигурация ---")
    print(f"Источник данных: {DATA_SOURCE}")
    print(f"Количество образцов: {N_SAMPLES}")
    print(f"Количество признаков: {N_FEATURES}")
    print(f"Количество классов: {N_CLASSES}")
    print(f"Размер батча: {BATCH_SIZE}")
    print(f"Эпохи: {EPOCHS}")
    print(f"Скорость обучения: {LEARNING_RATE}")
    print(f"Путь сохранения модели: {MODEL_PATH}")
    print(f"Имена классов: {class_names}")
    print("-" * 30)

    # 1. Генерация данных
    X, y = make_classification_data(n_samples=N_SAMPLES, n_features=N_FEATURES,
                                    n_classes=N_CLASSES, source=DATA_SOURCE)

    # 2. Создание датасета и даталоадера
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f'\nРазмер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # 3. Создание модели, функции потерь и оптимизатора
    model = LogisticRegression(in_features=N_FEATURES, num_classes=N_CLASSES)
    # nn.CrossEntropyLoss объединяет LogSoftmax и NLLLoss.
    # Она ожидает сырые логиты от модели и целочисленные метки (от 0 до C-1).
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print(f"\nАрхитектура модели:\n{model}")

    # 4. Обучение модели
    train_model(model, dataloader, criterion, optimizer, EPOCHS, N_CLASSES)

    # 5. Сохранение модели
    try:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"\nМодель сохранена в {MODEL_PATH}")
    except Exception as e:
        print(f"Ошибка при сохранении модели: {e}")

    # 6. Загрузка и оценка модели
    new_model = LogisticRegression(in_features=N_FEATURES, num_classes=N_CLASSES)
    if os.path.exists(MODEL_PATH):
        try:
            new_model.load_state_dict(torch.load(MODEL_PATH))
            print(f"Модель загружена из {MODEL_PATH}")
            evaluate_model(new_model, dataloader, N_CLASSES, class_names)
        except Exception as e:
            print(f"Ошибка при загрузке или оценке модели: {e}")
    else:
        print(f"Файл модели не найден по пути {MODEL_PATH}, пропуск загрузки и оценки.")


if __name__ == '__main__':
    main()
