import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)


class LinearRegression(nn.Module):
    """Базовая реализация линейной регрессии с одним выходом.

    Attributes:
        linear (nn.Linear): Полносвязный слой для линейного преобразования.
    """

    def __init__(self, in_features):
        """Инициализирует линейный слой.

        Args:
            in_features (int): Размерность входных признаков.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        """Прямой проход модели.

        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, in_features).

        Returns:
            torch.Tensor: Выходной тензор формы (batch_size, 1).
        """
        return self.linear(x)

    def get_weights(self):
        """Возвращает веса модели.

        Returns:
            torch.Tensor: Тензор весов формы (1, in_features).
        """
        return self.linear.weight

    def get_bias(self):
        """Возвращает смещение модели.

        Returns:
            torch.Tensor: Тензор смещения формы (1,).
        """
        return self.linear.bias


class RegularizedLinearRegression(LinearRegression):
    """Линейная регрессия с L1 и L2 регуляризацией.

    Attributes:
        l1_lambda (float): Коэффициент для L1 регуляризации.
        l2_lambda (float): Коэффициент для L2 регуляризации.
    """

    def __init__(self, in_features, l1_lambda=0.01, l2_lambda=0.01):
        """Инициализирует модель с параметрами регуляризации.

        Args:
            in_features (int): Размерность входных признаков.
            l1_lambda (float, optional): Коэффициент L1 регуляризации. По умолчанию 0.01.
            l2_lambda (float, optional): Коэффициент L2 регуляризации. По умолчанию 0.01.
        """
        super().__init__(in_features)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def regularization_loss(self):
        """Вычисляет потери от регуляризации.

        Returns:
            torch.Tensor: Суммарные потери от L1 и L2 регуляризации.
        """
        l1_loss = torch.norm(self.get_weights(), p=1) * self.l1_lambda
        l2_loss = torch.norm(self.get_weights(), p=2) ** 2 * self.l2_lambda
        return l1_loss + l2_loss


def train_with_early_stopping(model, dataloader, criterion, optimizer,
                              epochs=100, patience=5, min_delta=0.001):
    """Обучает модель с ранней остановкой при отсутствии улучшений."""
    best_loss = float('inf')
    no_improvement = 0
    losses = []  # Для хранения истории потерь

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)

            loss = criterion(y_pred, batch_y)
            if isinstance(model, RegularizedLinearRegression):
                loss += model.regularization_loss()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)
        losses.append(avg_loss)

        # Логирование
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

        # Early stopping проверка
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            no_improvement = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f'Early stopping at epoch {epoch}, best loss: {best_loss:.4f}')
            break

    # Загружаем лучшие веса
    model.load_state_dict(torch.load('models/best_model.pth'))

    # Сохраняем график потерь
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('plots/training_loss.png')
    plt.close()

    return model


class RegressionDataset(Dataset):
    """Датасет для задач регрессии."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_regression_data(n=100, noise=0.1, source='random'):
    """Генерация данных для регрессии.

    Args:
        n (int): Количество образцов
        noise (float): Уровень шума
        source (str): Источник данных ('random' или 'diabetes')
    """
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')


def mse(y_pred, y_true):
    """Вычисление среднеквадратичной ошибки.

    Args:
        y_pred: Предсказанные значения
        y_true: Истинные значения

    Returns:
        float: Значение MSE
    """
    return ((y_pred - y_true) ** 2).mean().item()


def log_epoch(epoch, loss, **metrics):
    """Логирование результатов эпохи.

    Args:
        epoch (int): Номер эпохи
        loss (float): Значение функции потерь
        metrics: Дополнительные метрики
    """
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)


def plot_regression_line(X, y, model, filename):
    """Визуализация данных и линии регрессии."""
    plt.figure(figsize=(10, 6))

    # Для многомерного случая показываем только первый признак
    if X.shape[1] > 1:
        print("Note: Showing only the first feature for visualization")

    # Сортируем данные для красивого отображения линии
    sorted_X, _ = torch.sort(X[:, 0])
    with torch.no_grad():
        predictions = model(sorted_X.unsqueeze(1))

    plt.scatter(X[:, 0].numpy(), y.numpy(), alpha=0.7, label='Data')
    plt.plot(sorted_X.numpy(), predictions.numpy(), 'r-', linewidth=3, label='Regression Line')

    plt.xlabel('Feature 1')
    plt.ylabel('Target')
    plt.title('Regression Line Visualization')
    plt.legend()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    # Генерируем данные
    X, y = make_regression_data(n=200)

    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # Создаём модель с регуляризацией
    model = RegularizedLinearRegression(in_features=1, l1_lambda=0.01, l2_lambda=0.01)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучаем модель с early stopping
    trained_model = train_with_early_stopping(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=100,
        patience=5,
        min_delta=0.001
    )

    # Сохраняем модель
    torch.save(trained_model.state_dict(), 'models/linreg_torch_regularized.pth')
    # Визуализации
    plot_regression_line(X, y, trained_model, 'plots/regression_line.png')
    # Загружаем модель
    new_model = RegularizedLinearRegression(in_features=1)
    new_model.load_state_dict(torch.load('models/linreg_torch_regularized.pth'))
    new_model.eval()
