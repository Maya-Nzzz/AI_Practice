import time

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import seaborn as sns
import os

from utils.experiment_utils import setup_experiment_logging, log_experiment_results, train_model
from utils.visualization_utils import plot_training_history, plot_heatmap, count_parameters
from utils.model_utils import FullyConnectedModel, prepare_classification_tensors, create_model_from_config

# Пути для сохранения результатов и графиков
RESULTS_PATH = "results/width_experiments"
PLOTS_PATH = "plots/width_experiments"
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


def run_width_experiments():
    """
    Запускает эксперименты с разной шириной слоев сети (ширина скрытых слоев),
    логирует результаты, сохраняет графики и метрики.
    """
    logger = setup_experiment_logging(RESULTS_PATH, "width_experiments")
    X_train, y_train, X_test, y_test = prepare_classification_tensors()
    # Создаем DataLoader'ы
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    widths = ['narrow', 'medium', 'wide', 'very_wide']

    results = {}
    for name in widths:
        logger.info(f"Training model: {name}")
        model = create_model_from_config(config_path=f'data/{name}.json')
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


def search_optimal_architecture():
    """
    Проводит поиск оптимальной архитектуры по трём стратегиям:
    - расширяющейся
    - сужающейся
    - с постоянной шириной

    Строит тепловые карты результатов и сохраняет лучшую конфигурацию по итоговой тестовой точности.
    """
    logger = setup_experiment_logging(RESULTS_PATH, "width_gridsearch")
    X_train, y_train, X_test, y_test = prepare_classification_tensors()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    strategies = ['expanding', 'shrinking', 'constant']

    # Задаём параметры сетки
    layer_counts = [2, 3, 4]
    base_sizes = [64, 128, 256]

    best_acc = 0
    best_config = None
    results = {}

    for strategy in strategies:
        heatmap_data = np.zeros((len(layer_counts), len(base_sizes)))

        for i, num_layers in enumerate(layer_counts):
            for j, base_size in enumerate(base_sizes):
                # Формируем список слоёв под стратегию
                if strategy == 'expanding':
                    sizes = [base_size * (l + 1) for l in range(num_layers)]
                elif strategy == 'shrinking':
                    sizes = [base_size * (num_layers - l) for l in range(num_layers)]
                elif strategy == 'constant':
                    sizes = [base_size] * num_layers
                else:
                    continue

                # Составляем конфиг "на лету"
                config = {
                    "input_dim": X_train.shape[1],
                    "output_dim": len(torch.unique(y_train)),
                    "layers": []
                }
                for size in sizes:
                    config["layers"].append({"type": "linear", "size": size})
                    config["layers"].append({"type": "relu"})

                # Создаём модель
                model = create_model_from_config(config_dict=config)
                history = train_model(model, train_loader, test_loader, logger=logger, epochs=5)

                final_acc = history['test_accs'][-1]
                heatmap_data[i, j] = final_acc

                # Сохраняем лучший результат
                if final_acc > best_acc:
                    best_acc = final_acc
                    best_config = config

        # Рисуем тепловую карту
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, xticklabels=base_sizes, yticklabels=layer_counts, cmap="viridis")
        plt.title(f"Strategy: {strategy} (test acc)")
        plt.xlabel("Base width")
        plt.ylabel("Num layers")
        plt.savefig(f"{PLOTS_PATH}/heatmap_{strategy}.png")
        plt.close()

        results[strategy] = heatmap_data

    print("Best accuracy:", best_acc)
    print("Best config:", best_config)

    logger.info(f"Best architecture: {best_config} with test accuracy {best_config:.4f}")


if __name__ == "__main__":
    run_width_experiments()
    search_optimal_architecture()
