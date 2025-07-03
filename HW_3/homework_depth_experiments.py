import os
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils.experiment_utils import setup_experiment_logging, log_experiment_results, train_model
from utils.visualization_utils import plot_training_history, count_parameters
from utils.model_utils import FullyConnectedModel

RESULTS_PATH = "results/depth_experiments"
PLOTS_PATH = "plots/depth_experiments"
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


def get_data(test_size=0.2, n_samples=2000, n_features=20, n_classes=2, random_state=42):
    """
    Генерирует синтетический датасет для классификации и возвращает готовые тензоры.
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                               n_informative=15, n_redundant=5, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))


def run_depth_experiments():
    """
    Запускает серию экспериментов с моделями разной глубины.
    Сохраняет кривые обучения и финальные результаты.
    """
    logger = setup_experiment_logging(RESULTS_PATH, "depth_experiments")

    X_train, y_train, X_test, y_test = get_data()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    depths = {
        '1_layer': [],
        '2_layers': [64],
        '3_layers': [128, 64],
        '5_layers': [256, 128, 64, 32],
        '7_layers': [512, 256, 128, 64, 32, 16]
    }

    results = {}

    for name, hidden in depths.items():
        logger.info(f"Training model: {name} (hidden={hidden})")

        model = FullyConnectedModel(config_path='data/config_example.json')

        start = time.time()
        history = train_model(model, train_loader, test_loader, logger=logger)
        elapsed = time.time() - start

        results[name] = {
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['test_accs'][-1],
            'history': history,
            'params': count_parameters(model),
            'time': elapsed
        }

        save_path = os.path.join(PLOTS_PATH, f"{name}_curve.png")
        plot_training_history(history, save_path=save_path, title=f"{name} learning curve")

        # Краткий финальный лог (детали уже залогированы в train_model)
        logger.info(
            f"{name}: Final train_acc={history['train_accs'][-1]:.4f}, "
            f"test_acc={history['test_accs'][-1]:.4f}, time={elapsed:.2f}s, params={results[name]['params']}"
        )

    log_experiment_results(logger, results)
    np.savez(f"{RESULTS_PATH}/depth_results.npz", **results)


def find_overfitting_epoch(history, patience=2):
    """
    Находит эпоху, когда test accuracy перестает расти (момент начала переобучения).
    """
    best = -float('inf')
    best_epoch = 0
    for i, acc in enumerate(history['test_accs']):
        if acc > best:
            best = acc
            best_epoch = i
        elif i - best_epoch >= patience:
            return best_epoch
    return len(history['test_accs']) - 1


def analyze_overfitting():
    """
    Проводит анализ переобучения для разных конфигураций.
    """
    logger = setup_experiment_logging(RESULTS_PATH, "overfitting_analysis")

    X_train, y_train, X_test, y_test = get_data()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    configs = [
        ("plain", {}),
        ("dropout", {"use_dropout": True, "dropout_p": 0.5}),
        ("batchnorm", {"use_batchnorm": True}),
        ("dropout+batchnorm", {"use_dropout": True, "dropout_p": 0.5, "use_batchnorm": True}),
    ]

    depths = {
        '2_layers': [64],
        '3_layers': [128, 64],
        '5_layers': [256, 128, 64, 32],
    }

    overfit_results = {}

    for name, hidden in depths.items():
        for cfg_name, model_kwargs in configs:
            logger.info(f"Analyzing overfitting: {name} + {cfg_name}")

            config_dict = {
                "input_dim": 20,
                "output_dim": 2,
                "layers": [{"type": "linear", "size": h} for h in hidden] + [{"type": "relu"}]
            }

            model = FullyConnectedModel(config_dict=config_dict)

            history = train_model(model, train_loader, test_loader, logger=logger)

            save_path = os.path.join(PLOTS_PATH, f"{name}_{cfg_name}_overfit.png")
            plot_training_history(history, save_path=save_path,
                                  title=f"{name} {cfg_name}")

            overfit_epoch = find_overfitting_epoch(history)

            overfit_results[f"{name}_{cfg_name}"] = {
                'final_train_acc': history['train_accs'][-1],
                'final_test_acc': history['test_accs'][-1],
                'overfit_epoch': overfit_epoch,
                'history': history
            }

            logger.info(
                f"{name} {cfg_name}: Final train_acc={history['train_accs'][-1]:.4f}, "
                f"test_acc={history['test_accs'][-1]:.4f}, overfit_epoch={overfit_epoch}"
            )

    best = max(overfit_results.items(), key=lambda x: x[1]['final_test_acc'])
    logger.info(f"Best config: {best[0]} with test accuracy {best[1]['final_test_acc']:.4f}")
    np.savez(f"{RESULTS_PATH}/overfitting_analysis.npz", **overfit_results)

    with open(f"{RESULTS_PATH}/best_depth.txt", "w") as f:
        f.write(f"Best config: {best[0]} with test accuracy {best[1]['final_test_acc']:.4f}\n")


if __name__ == "__main__":
    run_depth_experiments()
    analyze_overfitting()
