import torch
import torch.nn as nn
import json


class FullyConnectedModel(nn.Module):
    """
    Полностью-связная (fully connected) модель, конфигурируемая через JSON-файл или словарь.

    Атрибуты:
        config (dict): Конфигурация модели, содержащая описание слоёв и размерности.
        input_dim (int): Размерность входа.
        output_dim (int): Размерность выхода.
        net (nn.Sequential): Последовательно объединённые слои модели.
    """

    def __init__(self, config_path=None, config_dict=None):
        """
        Инициализация модели.

        Аргументы:
            config_path (str, optional): Путь к JSON-файлу с конфигурацией.
            config_dict (dict, optional): Словарь с конфигурацией.

        Исключения:
            ValueError: Если не передан ни config_path, ни config_dict.
            ValueError: Если в конфиге не указаны 'input_dim' и 'output_dim'.
        """
        super().__init__()

        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Нужно передать либо config_path, либо config_dict")

        self.input_dim = self.config.get("input_dim")
        self.output_dim = self.config.get("output_dim")
        layer_config = self.config.get("layers", [])

        if self.input_dim is None or self.output_dim is None:
            raise ValueError("В конфиге должны быть указаны 'input_dim' и 'output_dim'")

        layers = []
        prev_dim = self.input_dim

        for layer_spec in layer_config:
            layer_type = layer_spec["type"]

            if layer_type == "linear":
                out_dim = layer_spec["size"]
                layers.append(nn.Linear(prev_dim, out_dim))
                prev_dim = out_dim

            elif layer_type == "relu":
                layers.append(nn.ReLU())

            elif layer_type == "sigmoid":
                layers.append(nn.Sigmoid())

            elif layer_type == "tanh":
                layers.append(nn.Tanh())

            elif layer_type == "dropout":
                rate = layer_spec.get("rate", 0.5)
                layers.append(nn.Dropout(rate))

            elif layer_type == "batch_norm":
                layers.append(nn.BatchNorm1d(prev_dim))

            elif layer_type == "layer_norm":
                layers.append(nn.LayerNorm(prev_dim))

            else:
                raise ValueError(f"Неизвестный тип слоя: {layer_type}")

        # Добавляем финальный выходной слой
        layers.append(nn.Linear(prev_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Прямой проход по сети.

        Аргументы:
            x (torch.Tensor): Входной батч тензоров.

        Возвращает:
            torch.Tensor: Результат после прохождения через все слои сети.
        """
        x = x.view(x.size(0), -1)
        return self.net(x)


def create_model_from_config(config_path=None, config_dict=None):
    """
    Создаёт экземпляр FullyConnectedModel из конфигурационного файла или словаря.

    Аргументы:
        config_path (str, optional): Путь к JSON-файлу с конфигурацией.
        config_dict (dict, optional): Словарь с конфигурацией.

    Возвращает:
        FullyConnectedModel: Инициализированная модель.
    """
    return FullyConnectedModel(config_path=config_path, config_dict=config_dict)
