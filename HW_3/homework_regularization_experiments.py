import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.experiment_utils import setup_experiment_logging, log_experiment_results, train_model
from utils.visualization_utils import count_parameters, plot_training_history
from utils.model_utils import FullyConnectedModel

RESULTS_PATH = "homework/results/regularization_experiments"
PLOTS_PATH = "homework/plots/regularization_experiments"
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


def get_data(test_size=0.2, n_samples=2000, n_features=20, n_classes=2, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                               n_informative=15, n_redundant=5, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long),
            torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))


def run_regularization_experiments():
    logger = setup_experiment_logging(RESULTS_PATH, "regularization_experiments")
    X_train, y_train, X_test, y_test = get_data()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)
    configs = [
        ("no_reg", {}, 0.0),
        ("dropout_0.1", {"use_dropout": True, "dropout_p": 0.1}, 0.0),
        ("dropout_0.3", {"use_dropout": True, "dropout_p": 0.3}, 0.0),
        ("dropout_0.5", {"use_dropout": True, "dropout_p": 0.5}, 0.0),
        ("batchnorm", {"use_batchnorm": True}, 0.0),
        ("dropout+batchnorm", {"use_dropout": True, "dropout_p": 0.5, "use_batchnorm": True}, 0.0),
        ("l2", {}, 1e-2),
    ]
    results = {}
    for name, model_kwargs, wd in configs:
        logger.info(f"Training model: {name}")
        model = FullyConnectedModel(config_path='data/config_example.json')
        history = train_model(model, train_loader, test_loader, weight_decay=wd, logger=logger)
        results[name] = {
            'final_train_acc': history['train'][-1],
            'final_test_acc': history['test'][-1],
            'history': history,
            'params': count_parameters(model)
        }
        plot_training_history(history, save_path=f"{PLOTS_PATH}/{name}_curve.png", title=f"{name} learning curve")
        # Визуализация распределения весов
        weights = torch.cat([p.data.flatten().cpu() for p in model.parameters() if p.requires_grad])
        plt.figure(figsize=(6, 4))
        plt.hist(weights.numpy(), bins=50)
        plt.title(f"Weight distribution: {name}")
        plt.savefig(f"{PLOTS_PATH}/{name}_weights.png")
        plt.close()
        logger.info(f"{name}: train_acc={history['train'][-1]:.4f}, test_acc={history['test'][-1]:.4f}")
    log_experiment_results(logger, results)
    np.savez(f"{RESULTS_PATH}/regularization_results.npz", **results)


def adaptive_regularization():
    logger = setup_experiment_logging(RESULTS_PATH, "adaptive_regularization")
    X_train, y_train, X_test, y_test = get_data()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    # Dropout с изменяющимся коэффициентом
    class AdaptiveDropoutNet(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_layers, dropout_start=0.5, dropout_end=0.1,
                     use_batchnorm=False):
            super().__init__()
            layers = []
            prev_dim = input_dim
            n_layers = len(hidden_layers)
            for i, h in enumerate(hidden_layers):
                layers.append(nn.Linear(prev_dim, h))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                p = dropout_start + (dropout_end - dropout_start) * (i / max(1, n_layers - 1))
                layers.append(nn.Dropout(p))
                prev_dim = h
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # BatchNorm с разным momentum
    class CustomBatchNormNet(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_layers, momentums):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for i, h in enumerate(hidden_layers):
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.BatchNorm1d(h, momentum=momentums[i]))
                layers.append(nn.ReLU())
                prev_dim = h
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # Адаптивный Dropout
    logger.info("Adaptive Dropout experiment")
    model = AdaptiveDropoutNet(20, 2, [128, 64, 32], dropout_start=0.5, dropout_end=0.1)
    history = train_model(model, train_loader, test_loader, logger=logger)
    plot_training_history(history, save_path=f"{PLOTS_PATH}/adaptive_dropout_curve.png", title="Adaptive Dropout")
    # BatchNorm с разными momentum
    logger.info("BatchNorm momentum experiment")
    model = CustomBatchNormNet(20, 2, [128, 64, 32], momentums=[0.1, 0.5, 0.9])
    history = train_model(model, train_loader, test_loader, logger=logger)
    plot_training_history(history, save_path=f"{PLOTS_PATH}/batchnorm_momentum_curve.png", title="BatchNorm Momentum")
    # Комбинирование техник
    logger.info("Combined techniques experiment")
    model = AdaptiveDropoutNet(20, 2, [128, 64, 32], dropout_start=0.5, dropout_end=0.1, use_batchnorm=True)
    history = train_model(model, train_loader, test_loader, logger=logger)
    plot_training_history(history, save_path=f"{PLOTS_PATH}/combined_adaptive_curve.png", title="Combined Adaptive")


if __name__ == "__main__":
    run_regularization_experiments()
    adaptive_regularization()
