import os
import time
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils.experiment_utils import setup_experiment_logging, log_experiment_results, train_model
from utils.visualization_utils import plot_training_history, count_parameters
from utils.model_utils import FullyConnectedModel, prepare_classification_tensors, create_model_from_config

RESULTS_PATH = "results/depth_experiments"
PLOTS_PATH = "plots/depth_experiments"
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


def run_depth_experiments():
    """
    Запускает серию экспериментов с моделями разной глубины.
    Сохраняет кривые обучения и финальные результаты.
    """
    logger = setup_experiment_logging(RESULTS_PATH, "depth_experiments")

    X_train, y_train, X_test, y_test = prepare_classification_tensors()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    depths = ['1_layer', '2_layers', '3_layers', '5_layers', '7_layers']

    results = {}

    for name in depths:
        model = create_model_from_config(config_path=f'data/{name}.json')

        logger.info(f"Training model: {name}")

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


if __name__ == "__main__":
    run_depth_experiments()
