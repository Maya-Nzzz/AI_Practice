import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

from utils.experiment_utils import setup_experiment_logging, log_experiment_results, train_model
from utils.visualization_utils import plot_training_history, plot_heatmap, count_parameters
from utils.model_utils import FullyConnectedModel

# Пути для сохранения результатов и графиков
RESULTS_PATH = "results/width_experiments"
PLOTS_PATH = "plots/width_experiments"
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


def get_data(test_size=0.2, n_samples=2000, n_features=20, n_classes=2, random_state=42):
    """
    Генерирует синтетические данные для задачи классификации и стандартизирует их.

    Args:
        test_size (float): Доля данных для теста.
        n_samples (int): Количество выборок.
        n_features (int): Количество признаков.
        n_classes (int): Количество классов.
        random_state (int): Фиксированное зерно генератора.

    Returns:
        Tuple[torch.Tensor]: Обучающие и тестовые данные (X_train, y_train, X_test, y_test).
    """
    # Создаем синтетический датасет
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                               n_informative=15, n_redundant=5, random_state=random_state)
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Нормализация признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))


def run_width_experiments():
    """
    Запускает эксперименты с разной шириной слоев сети (ширина скрытых слоев),
    логирует результаты, сохраняет графики и метрики.
    """
    logger = setup_experiment_logging(RESULTS_PATH, "width_experiments")
    X_train, y_train, X_test, y_test = get_data()
    # Создаем DataLoader'ы
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    # Конфигурации ширины скрытых слоев
    widths = {
        'narrow': [64, 32, 16],
        'medium': [256, 128, 64],
        'wide': [1024, 512, 256],
        'very_wide': [2048, 1024, 512]
    }

    results = {}
    for name, hidden in widths.items():
        logger.info(f"Training model: {name} (hidden={hidden})")
        # Загружаем модель из конфига
        model = FullyConnectedModel(config_path='data/config_example.json')
        start = time.time()
        # Запускаем обучение
        history = train_model(model, train_loader, test_loader, logger=logger)
        elapsed = time.time() - start

        # Подсчет числа параметров
        params_count = count_parameters(model)

        # Сохраняем метрики
        results[name] = {
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['test_accs'][-1],
            'params': params_count,
            'time': elapsed
        }

        # Сохраняем график обучения
        plot_training_history(history, save_path=f"{PLOTS_PATH}/{name}_curve.png", title=f"{name} learning curve")
        logger.info(
            f"{name}: train_acc={history['train_accs'][-1]:.4f}, test_acc={history['test_accs'][-1]:.4f}, time={elapsed:.2f}s, params={params_count}")

    # Логируем финальные результаты и сохраняем
    log_experiment_results(logger, results)
    np.savez(f"{RESULTS_PATH}/width_results.npz", **results)


def optimize_architecture():
    """
    Проводит перебор архитектур по схемам (расширяющаяся, сужающаяся, постоянная ширина),
    строит тепловые карты и сохраняет конфигурацию с лучшей точностью.
    """
    logger = setup_experiment_logging(RESULTS_PATH, "width_gridsearch")
    X_train, y_train, X_test, y_test = get_data()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    layer_sizes = [16, 32, 64, 128, 256]

    # Определение схем генерации конфигураций
    schemes = {
        'expanding': lambda: [(l, l * 2, l * 4) for l in layer_sizes if l * 4 <= 256],
        'shrinking': lambda: [(l * 4, l * 2, l) for l in layer_sizes if l * 4 <= 256],
        'constant': lambda: [(l, l, l) for l in layer_sizes]
    }

    all_results = {}
    for scheme, gen in schemes.items():
        for hidden in gen():
            model = FullyConnectedModel(config_path='data/config_example.json')
            history = train_model(model, train_loader, test_loader, logger=logger, epochs=10)
            all_results[(scheme, hidden)] = history['test_accs'][-1]

    # Генерация тепловых карт по каждой схеме
    for scheme in schemes:
        data = np.zeros((len(layer_sizes), len(layer_sizes)))
        for i, l1 in enumerate(layer_sizes):
            for j, l2 in enumerate(layer_sizes):
                # Проверяем, подходит ли конфигурация под схему
                if scheme == 'expanding' and l2 == l1 * 2:
                    hidden = (l1, l2, l2 * 2)
                elif scheme == 'shrinking' and l1 == l2 * 2:
                    hidden = (l1, l2, l2 // 2)
                elif scheme == 'constant' and l1 == l2:
                    hidden = (l1, l2, l2)
                else:
                    continue

                if (scheme, hidden) in all_results:
                    data[i, j] = all_results[(scheme, hidden)]

        plot_heatmap(data, x_labels=layer_sizes, y_labels=layer_sizes,
                     save_path=f"{PLOTS_PATH}/width_heatmap_{scheme}.png", title=f"Grid Search {scheme}")

    # Поиск лучшей конфигурации
    best = max(all_results.items(), key=lambda x: x[1])
    logger.info(f"Best width config: {best[0]} with test accuracy {best[1]:.4f}")
    with open(f"{RESULTS_PATH}/best_width.txt", "w") as f:
        f.write(f"Best width config: {best[0]} with test accuracy {best[1]:.4f}\n")


if __name__ == "__main__":
    run_width_experiments()
    optimize_architecture()
