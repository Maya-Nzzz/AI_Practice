import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt

from utils.experiment_utils import setup_experiment_logging, log_experiment_results, train_model
from utils.visualization_utils import plot_training_history, count_parameters

from utils.model_utils import create_model_from_config, prepare_classification_tensors

RESULTS_PATH = "results/regularization_experiments"
PLOTS_PATH = "plots/regularization_experiments"
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


def run_regularization_experiments():
    logger = setup_experiment_logging(RESULTS_PATH, "regularization_experiments")
    X_train, y_train, X_test, y_test = prepare_classification_tensors()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    configs = [
        ("no_reg", 'data/config_no_reg.json', 0.0),
        ("dropout_0.1", 'data/config_dropout_0.1.json', 0.0),
        ("dropout_0.3", 'data/config_dropout_0.3.json', 0.0),
        ("dropout_0.5", 'data/config_dropout_0.5.json', 0.0),
        ("batchnorm", 'data/config_batchnorm.json', 0.0),
        ("dropout+batchnorm", 'data/config_dropout_batchnorm.json', 0.0),
        ("l2", 'data/config_no_reg.json', 1e-2),  # используем ту же архитектуру, но с L2
    ]
    results = {}
    for name, config_path, wd in configs:
        logger.info(f"Training model: {name}")
        model = create_model_from_config(config_path=config_path)
        history = train_model(model, train_loader, test_loader, logger=logger, weight_decay=wd)
        results[name] = {
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['train_accs'][-1],
            'history': history,
            'params': count_parameters(model)
        }
        plot_training_history(history, save_path=f"{PLOTS_PATH}/{name}_curve.png", title=f"{name} learning curve")

        # Визуализация распределения весов
        weights = torch.cat([p.data.flatten().cpu() for p in model.parameters() if p.requires_grad])
        plt.figure(figsize=(6, 4))
        plt.hist(weights.numpy(), bins=50)
        plt.title(f"Распределение весов: {name}")
        plt.savefig(f"{PLOTS_PATH}/{name}_weights.png")
        plt.close()
        logger.info(f"{name}: train_acc={history['train_accs'][-1]:.4f}, test_acc={history['train_accs'][-1]:.4f}")

    log_experiment_results(logger, results)
    np.savez(f"{RESULTS_PATH}/regularization_results.npz", **results)


def adaptive_regularization():
    logger = setup_experiment_logging(RESULTS_PATH, "adaptive_regularization")
    X_train, y_train, X_test, y_test = prepare_classification_tensors()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    # Адаптивный Dropout: подготавливаем конфиг
    adaptive_dropout_config = {
        "input_dim": 20,
        "output_dim": 2,
        "layers": [
            {"type": "linear", "size": 128},
            {"type": "batch_norm"},
            {"type": "relu"},
            {"type": "dropout", "rate": 0.5},
            {"type": "linear", "size": 64},
            {"type": "batch_norm"},
            {"type": "relu"},
            {"type": "dropout", "rate": 0.3},
            {"type": "linear", "size": 32},
            {"type": "batch_norm"},
            {"type": "relu"},
            {"type": "dropout", "rate": 0.1}
        ]
    }
    logger.info("Adaptive Dropout experiment")
    model = create_model_from_config(config_dict=adaptive_dropout_config)
    history = train_model(model, train_loader, test_loader, logger=logger)
    plot_training_history(history, save_path=f"{PLOTS_PATH}/adaptive_dropout_curve.png", title="Адаптивная регуляризация")

    batchnorm_config = {
        "input_dim": 20,
        "output_dim": 2,
        "layers": [
            {"type": "linear", "size": 128},
            {"type": "batch_norm"},
            {"type": "relu"},
            {"type": "linear", "size": 64},
            {"type": "batch_norm"},
            {"type": "relu"},
            {"type": "linear", "size": 32},
            {"type": "batch_norm"},
            {"type": "relu"}
        ]
    }
    logger.info("BatchNorm experiment")
    model = create_model_from_config(config_dict=batchnorm_config)
    history = train_model(model, train_loader, test_loader, logger=logger)
    plot_training_history(history, save_path=f"{PLOTS_PATH}/batchnorm_momentum_curve.png", title="BatchNorm")

    # Комбинированный конфиг
    combined_config = adaptive_dropout_config  # Можно использовать тот же конфиг
    logger.info("Combined techniques experiment")
    model = create_model_from_config(config_dict=combined_config)
    history = train_model(model, train_loader, test_loader, logger=logger)
    plot_training_history(history, save_path=f"{PLOTS_PATH}/combined_adaptive_curve.png", title="Combined Adaptive")


if __name__ == "__main__":
    run_regularization_experiments()
    adaptive_regularization()
