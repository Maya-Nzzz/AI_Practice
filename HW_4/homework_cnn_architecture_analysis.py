import os
import pprint
import time
import torch

from HW_4.homework_cnn_vs_fc_comparison import load_dataset
from utils.training_utils import train_model, calculate_accuracy
from utils.visualization_utils import count_parameters, plot_training_history, visualize_activations
from models.cnn_models import MultiKernelConvNet, Adjustable_Depth_CNN, CNNWithResidual

PLOTS_PATH = "plots/cnn_architecture"
os.makedirs(PLOTS_PATH, exist_ok=True)


def get_impact_kernel_size(device):
    train_loader, test_loader = load_dataset('cifar', 128)
    epochs = 10
    kernels_list = [[3, 3], [5, 5], [7, 7], [1, 3]]
    results = {}
    for ks in kernels_list:
        name = f"Kernels_{'x'.join(map(str, ks))}"
        print(f"\n▶ Запуск эксперимента: {name}")
        model = MultiKernelConvNet(3, 10, ks).to(device)
        params = count_parameters(model)

        rf = 1 + sum(k - 1 for k in ks)  # рецептивное поле

        start = time.time()
        history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
        training_time = time.time() - start

        test_acc = calculate_accuracy(test_loader, model, device)
        plot_training_history(history, epochs, f'{PLOTS_PATH}/{name}.png', title=f"{name} Кривые обучения")

        visualize_activations(
            model.feature_extractor[0], next(iter(test_loader))[0].to(device),
            f"{PLOTS_PATH}/First-Layer_activations_{name}",
            chart_title=f"Активации первого слоя: {name}"
        )

        # сохраняем результаты
        results[name] = {'params': params, 'rf': rf, 'test_acc': test_acc, 'training_time': training_time}

    for name, data in results.items():
        print(f'''{name}: время обучения = {data["training_time"]:.2f} сек, точность на тесте = {data["test_acc"]:.4f}
                      количество параметров = {data["params"]}, рецептивное поле = {data["rf"]}''')

    return None


def get_impact_depth(device):
    train_loader, test_loader = load_dataset('cifar', 128)
    epochs = 10
    depths = [2, 4, 6]
    results = {}
    for d in depths:
        name = f"Depth_{d}"
        print(f"\n▶ Запуск эксперимента: {name}")
        model = Adjustable_Depth_CNN(3, 10, d).to(device)

        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
        training_time = time.time() - start_time

        train_acc = calculate_accuracy(train_loader, model, device)
        test_acc = calculate_accuracy(test_loader, model, device)

        plot_training_history(history, epochs, f'{PLOTS_PATH}/{name}.png', title=f"{name} Кривые обучения")
        results[name] = {'train_acc': train_acc, 'test_acc': test_acc, 'training_time': training_time}

    print("\n▶ Запуск эксперимента: CNN с Residual-блоками")
    model_with_residual = CNNWithResidual(input_channels=3).to(device)

    start_time = time.time()
    history = train_model(model_with_residual, train_loader, test_loader, epochs=10, device=device)
    training_time = time.time() - start_time

    train_acc = calculate_accuracy(train_loader, model_with_residual, device)
    test_acc = calculate_accuracy(test_loader, model_with_residual, device)

    plot_training_history(history, 10, f'{PLOTS_PATH}/CNNWithResidual.png', title="CNNWithResidual Кривые обучения")
    results['CNNWithResidual'] = {'train_acc': train_acc, 'test_acc': test_acc, 'training_time': training_time}

    for name, data in results.items():
        print(f'{name}: время обучения = {data["training_time"]:.2f} сек, точность на тесте = {data["test_acc"]:.4f}')
    return None


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + '=' * 30,
          "\nАнализ влияния размера ядра",
          "\n" + '=' * 30)
    get_impact_kernel_size(device)

    print("\n" + '=' * 30,
          "\nАнализ влияния глубины сети",
          "\n" + '=' * 30)
    get_impact_depth(device)
