import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from HW_4.models.cnn_models import SimpleCNN, CNNWithResidual, CIFARCNN
from HW_4.models.fc_models import FullyConnectedModel
from HW_4.utils.training_utils import train_model
from HW_4.utils.visualization_utils import count_parameters, plot_training_history, plot_confusion_matrix

PLOTS_PATH = "plots/cnn_vs_fc_comparison"
os.makedirs(PLOTS_PATH, exist_ok=True)


def load_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        training_dataset = datasets.MNIST(root='data', train=True, download=True, transform=data_transforms)
        testing_dataset = datasets.MNIST(root='data', train=False, download=True, transform=data_transforms)
    else:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        training_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=data_transforms)
        testing_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=data_transforms)

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return training_loader, testing_loader


def mnist_comparison(device):
    batch_size = 128
    models = {
        'FullyConnectedModel': FullyConnectedModel(),
        'SimpleCNN': SimpleCNN(),
        'CNNWithResidual': CNNWithResidual()
    }
    train_loader, test_loader = load_dataset('mnist', batch_size)
    results = {}
    for name, model in models.items():
        print(f"\n===Обучение {name} на MNIST===")
        model.to(device)

        start_time = time.time()
        history = train_model(model, train_loader, test_loader)
        training_time = time.time() - start_time

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, preds = torch.max(output, 1)
                correct += (preds == target).sum().item()
                total += target.size(0)
        test_acc = correct / total

        params = count_parameters(model)

        plot_training_history(history, 5, f'{PLOTS_PATH}/{name}_on_MNIST.png', title=f"{name} Learning Curves (MNIST)")

        results[name] = {
            'train_time': training_time,
            'test_accuracy': test_acc,
            'num_params': params,
            'history': history
        }

    for name, data in results.items():
        print(
            f'{name}: train_time = {data["train_time"]}, test_accuracy = {data["test_accuracy"]}, num_params = {data["num_params"]}')

    return results


def cifar_comparison(device):
    batch_size = 128
    epochs = 10
    models = {
        'FullyConnectedModel': FullyConnectedModel(version='deep'),
        'CIFARCNN': CIFARCNN(),
        'CNNWithResidual': CNNWithResidual(input_channels=3)
    }
    train_loader, test_loader = load_dataset('cifar', batch_size)

    results = {}
    for name, model in models.items():
        print(f"\n===Обучение {name} на CIFAR-10===")
        model.to(device)

        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=epochs)
        training_time = time.time() - start_time

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, preds = torch.max(output, 1)
                correct += (preds == target).sum().item()
                total += target.size(0)
        test_acc = correct / total

        plot_confusion_matrix(
            model, f'{PLOTS_PATH}/confusion_matrix_{name}.png', test_loader, device,
            classes=test_loader.dataset.classes,
            title=f"{name} Confusion Matrix (CIFAR-10)"
        )

        results[name] = {
            'train_time': training_time,
            'test_accuracy': test_acc,
            'history': history
        }

    for name, data in results.items():
        print(
            f'{name}: train_time = {data["train_time"]}, test_accuracy = {data["test_accuracy"]}')
    return results


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + '=' * 30,
          "\nСравнение на MNIST",
          "\n" + '=' * 30)
    mnist_comparison(device)

    print("\n" + '=' * 30,
          "\nСравнение на CIFAR-10",
          "\n" + '=' * 30)
    cifar_comparison(device)
