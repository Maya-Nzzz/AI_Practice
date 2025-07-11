import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from HW_4.utils.visualization_utils import count_parameters


class CustomConv2d(nn.Module):
    """
    Сверточный слой с дополнительной логикой нормализации активаций после свертки.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        # Доп. логика: нормализация по каналу
        x = x - x.mean(dim=(2, 3), keepdim=True)
        return x


class SpatialAttention(nn.Module):
    """
    Простой spatial attention слой: вычисляет карту важности для каждого пикселя.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attention_map = torch.sigmoid(self.conv(x))
        return x * attention_map


class CustomActivation(Function):
    """
    Пример: сглаженный ReLU, который в отрицательной части растёт экспоненциально.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.where(input > 0, input, torch.exp(input) - 1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.where(input > 0, torch.ones_like(input), torch.exp(input))
        return grad_output * grad_input


class CustomActivationLayer(nn.Module):
    def forward(self, x):
        return CustomActivation.apply(x)


class CustomMaxAvgPool2d(nn.Module):
    """
    Пуллинг, который берет среднее между MaxPool и AvgPool результатов.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        max_pooled = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        avg_pooled = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return 0.5 * (max_pooled + avg_pooled)


class BasicResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return F.relu(out)


class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, widening_factor=2):
        super().__init__()
        widened_channels = in_channels * widening_factor
        self.conv1 = nn.Conv2d(in_channels, widened_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widened_channels)
        self.conv2 = nn.Conv2d(widened_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


def experiment_with_custom_layers(x):
    # Test CustomConv2d
    conv = CustomConv2d(3, 8, kernel_size=3, padding=1)
    y_conv = conv(x)
    print("CustomConv2d output shape:", y_conv.shape)

    # Test SpatialAttention
    att = SpatialAttention(8)
    y_att = att(y_conv)
    print("SpatialAttention output shape:", y_att.shape)

    # Test CustomActivation
    act = CustomActivationLayer()
    x_act = torch.randn(2, 2, requires_grad=True)
    y_act = act(x_act)
    print("CustomActivation output:", y_act)

    # Test backward for CustomActivation
    y_act.sum().backward()
    print("CustomActivation grad:", x_act.grad)

    # Test CustomMaxAvgPool2d
    pool = CustomMaxAvgPool2d(kernel_size=2)
    y_pool = pool(x)
    print("CustomMaxAvgPool2d output shape:", y_pool.shape)


def experiments_with_residual_blocks(x):
    # Базовый блок
    basic_block = BasicResidualBlock(64)
    y_basic = basic_block(x)
    print("Basic block output shape:", y_basic.shape)
    print("Basic block params:", count_parameters(basic_block))
    # Bottleneck блок
    bottleneck_block = BottleneckResidualBlock(64, 16)
    y_bottle = bottleneck_block(x)
    print("Bottleneck block output shape:", y_bottle.shape)
    print("Bottleneck block params:", count_parameters(bottleneck_block))
    # Wide блок
    wide_block = WideResidualBlock(64, widening_factor=4)
    y_wide = wide_block(x)
    print("Wide block output shape:", y_wide.shape)
    print("Wide block params:", count_parameters(wide_block))


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn(1, 3, 32, 32)
    experiment_with_custom_layers(x)

    x = torch.randn(1, 64, 32, 32)
    experiments_with_residual_blocks(x)
