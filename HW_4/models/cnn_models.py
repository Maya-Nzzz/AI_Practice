import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block с двумя conv-слоями, shortcut формируется внутри при необходимости.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        return self.relu(out)


class CNNWithResidual(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, stride=2)
        self.res3 = ResidualBlock(64, 64)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CIFARCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiKernelConvNet(nn.Module):
    def __init__(self, input_channels, class_count, kernel_sizes):
        super().__init__()
        conv_blocks = []
        current_channels = input_channels

        # Для каждого ядра создаём блок
        for kernel_size in kernel_sizes:
            conv_layer = nn.Conv2d(
                in_channels=current_channels,
                out_channels=64,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False
            )
            norm_layer = nn.BatchNorm2d(64)
            activation = nn.ReLU()

            # Добавляем блок в список
            conv_blocks.extend([conv_layer, norm_layer, activation])
            current_channels = 64

        # Добавляем адаптивное усреднение и финальную классификацию
        final_pool = nn.AdaptiveAvgPool2d(output_size=1)
        flatten_layer = nn.Flatten()
        fc_layer = nn.Linear(64, class_count)

        # Объединяем все слои в последовательность
        self.feature_extractor = nn.Sequential(*conv_blocks, final_pool, flatten_layer)
        self.classifier = fc_layer

    def forward(self, input_tensor):
        """
        Выполняем прямое распространение через свёрточные блоки,
        затем классификацию.
        """
        features = self.feature_extractor(input_tensor)
        output = self.classifier(features)
        return output


class Adjustable_Depth_CNN(nn.Module):
    def __init__(self, input_channels, output_classes, hidden_layers_count):
        super().__init__()
        components = []
        current_channels = input_channels
        # Добавление указанного количества сверточных блоков
        for _ in range(hidden_layers_count):
            components.append(nn.Conv2d(current_channels, 32, kernel_size=3, padding=1))
            components.append(nn.ReLU())
            current_channels = 32
        # Завершающие слои для классификации
        components.append(nn.AdaptiveAvgPool2d(1))
        components.append(nn.Flatten())
        components.append(nn.Linear(32, output_classes))
        self.model_sequence = nn.Sequential(*components)

    def forward(self, input_tensor):
        return self.model_sequence(input_tensor)