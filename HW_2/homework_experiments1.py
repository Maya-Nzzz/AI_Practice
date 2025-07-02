import matplotlib.pyplot as plt
from torch.optim import SGD, Adam, RMSprop
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

from HW_2.homework_datasets import train_loader_mpg, val_loader_mpg, input_dim_mpg, LinearRegressionModel, \
    train_loader_diabetes, val_loader_diabetes, input_dim_diabetes, LogisticRegressionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Модифицированная функция обучения для сбора метрик
def train_model_with_metrics(model, train_loader, val_loader, criterion, optimizer, num_epochs, task_type):
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if outputs.shape[-1] == 1 and labels.dim() == 1:
                labels = labels.unsqueeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if task_type == 'classification':
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            else:
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)

        # Валидация
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                if outputs.shape[-1] == 1 and labels.dim() == 1:
                    labels = labels.unsqueeze(1)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                if task_type == 'classification':
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                else:
                    val_preds.extend(outputs.detach().cpu().numpy())
                    val_labels.extend(labels.detach().cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        metrics['val_loss'].append(avg_val_loss)

        # Расчет метрик
        if task_type == 'regression':
            train_rmse = np.sqrt(mean_squared_error(train_labels, train_preds))
            val_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
            metrics['train_rmse'].append(train_rmse)
            metrics['val_rmse'].append(val_rmse)
        else:
            train_acc = accuracy_score(train_labels, train_preds)
            val_acc = accuracy_score(val_labels, val_preds)
            train_f1 = f1_score(train_labels, train_preds)
            val_f1 = f1_score(val_labels, val_preds)
            metrics['train_acc'].append(train_acc)
            metrics['val_acc'].append(val_acc)
            metrics['train_f1'].append(train_f1)
            metrics['val_f1'].append(val_f1)

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return metrics


# Функция для экспериментов с гиперпараметрами
def run_experiments(dataset_type='regression'):
    if dataset_type == 'regression':
        train_loader = train_loader_mpg
        val_loader = val_loader_mpg
        input_dim = input_dim_mpg
        model_class = LinearRegressionModel
        criterion = nn.MSELoss()
        task_type = 'regression'
    else:
        train_loader = train_loader_diabetes
        val_loader = val_loader_diabetes
        input_dim = input_dim_diabetes
        model_class = LogisticRegressionModel
        criterion = nn.BCEWithLogitsLoss()
        task_type = 'classification'

    # Параметры для экспериментов
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    optimizers = {
        'SGD': SGD,
        'Adam': Adam,
        'RMSprop': RMSprop
    }

    results = []

    # Эксперименты с разными learning rate
    print("\nЭксперименты с разными learning rate (batch_size=32, Adam)")
    for lr in learning_rates:
        model = model_class(input_dim).to(device)
        optimizer = Adam(model.parameters(), lr=lr)

        print(f"\nLearning rate: {lr}")
        metrics = train_model_with_metrics(
            model, train_loader, val_loader, criterion,
            optimizer, num_epochs=50, task_type=task_type
        )

        best_val_loss = min(metrics['val_loss'])
        if task_type == 'regression':
            best_val_rmse = min(metrics['val_rmse'])
            results.append({
                'type': 'learning_rate',
                'param': lr,
                'val_loss': best_val_loss,
                'val_rmse': best_val_rmse
            })
            print(f"Best val loss: {best_val_loss:.4f}, Best val RMSE: {best_val_rmse:.4f}")
        else:
            best_val_acc = max(metrics['val_acc'])
            best_val_f1 = max(metrics['val_f1'])
            results.append({
                'type': 'learning_rate',
                'param': lr,
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'val_f1': best_val_f1
            })
            print(
                f"Best val loss: {best_val_loss:.4f}, Best val acc: {best_val_acc:.4f}, Best val F1: {best_val_f1:.4f}")

    # Эксперименты с разными batch sizes
    print("\nЭксперименты с разными batch sizes (lr=0.01, Adam)")
    for batch_size in batch_sizes:
        # Создаем новые DataLoader'ы с текущим batch_size
        train_loader_bs = DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader_bs = DataLoader(
            val_loader.dataset,
            batch_size=batch_size,
            shuffle=False
        )

        model = model_class(input_dim).to(device)
        optimizer = Adam(model.parameters(), lr=0.01)

        print(f"\nBatch size: {batch_size}")
        metrics = train_model_with_metrics(
            model, train_loader_bs, val_loader_bs, criterion,
            optimizer, num_epochs=50, task_type=task_type
        )

        best_val_loss = min(metrics['val_loss'])
        if task_type == 'regression':
            best_val_rmse = min(metrics['val_rmse'])
            results.append({
                'type': 'batch_size',
                'param': batch_size,
                'val_loss': best_val_loss,
                'val_rmse': best_val_rmse
            })
            print(f"Best val loss: {best_val_loss:.4f}, Best val RMSE: {best_val_rmse:.4f}")
        else:
            best_val_acc = max(metrics['val_acc'])
            best_val_f1 = max(metrics['val_f1'])
            results.append({
                'type': 'batch_size',
                'param': batch_size,
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'val_f1': best_val_f1
            })
            print(
                f"Best val loss: {best_val_loss:.4f}, Best val acc: {best_val_acc:.4f}, Best val F1: {best_val_f1:.4f}")

    # Эксперименты с разными оптимизаторами
    print("\nЭксперименты с разными оптимизаторами (lr=0.01, batch_size=32)")
    for opt_name, opt_class in optimizers.items():
        model = model_class(input_dim).to(device)
        optimizer = opt_class(model.parameters(), lr=0.01)

        print(f"\nOptimizer: {opt_name}")
        metrics = train_model_with_metrics(
            model, train_loader, val_loader, criterion,
            optimizer, num_epochs=50, task_type=task_type
        )

        best_val_loss = min(metrics['val_loss'])
        if task_type == 'regression':
            best_val_rmse = min(metrics['val_rmse'])
            results.append({
                'type': 'optimizer',
                'param': opt_name,
                'val_loss': best_val_loss,
                'val_rmse': best_val_rmse
            })
            print(f"Best val loss: {best_val_loss:.4f}, Best val RMSE: {best_val_rmse:.4f}")
        else:
            best_val_acc = max(metrics['val_acc'])
            best_val_f1 = max(metrics['val_f1'])
            results.append({
                'type': 'optimizer',
                'param': opt_name,
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'val_f1': best_val_f1
            })
            print(
                f"Best val loss: {best_val_loss:.4f}, Best val acc: {best_val_acc:.4f}, Best val F1: {best_val_f1:.4f}")

    return results


# Функция для визуализации результатов
def visualize_results(results, task_type):
    plt.figure(figsize=(15, 5))

    # Разделяем результаты по типам экспериментов
    lr_results = [r for r in results if r['type'] == 'learning_rate']
    bs_results = [r for r in results if r['type'] == 'batch_size']
    opt_results = [r for r in results if r['type'] == 'optimizer']

    # График для learning rate
    plt.subplot(1, 3, 1)
    if task_type == 'regression':
        plt.plot([r['param'] for r in lr_results], [r['val_rmse'] for r in lr_results], 'o-')
        plt.ylabel('Validation RMSE')
    else:
        plt.plot([r['param'] for r in lr_results], [r['val_acc'] for r in lr_results], 'o-')
        plt.ylabel('Validation Accuracy')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.title('Learning Rate Experiments')

    # График для batch size
    plt.subplot(1, 3, 2)
    if task_type == 'regression':
        plt.plot([r['param'] for r in bs_results], [r['val_rmse'] for r in bs_results], 'o-')
        plt.ylabel('Validation RMSE')
    else:
        plt.plot([r['param'] for r in bs_results], [r['val_acc'] for r in bs_results], 'o-')
        plt.ylabel('Validation Accuracy')
    plt.xlabel('Batch Size')
    plt.title('Batch Size Experiments')

    # График для оптимизаторов
    plt.subplot(1, 3, 3)
    if task_type == 'regression':
        plt.bar([r['param'] for r in opt_results], [r['val_rmse'] for r in opt_results])
        plt.ylabel('Validation RMSE')
    else:
        plt.bar([r['param'] for r in opt_results], [r['val_acc'] for r in opt_results])
        plt.ylabel('Validation Accuracy')
    plt.xlabel('Optimizer')
    plt.title('Optimizer Experiments')

    plt.tight_layout()
    plt.show()

    # Вывод таблицы с результатами
    print("\nСводная таблица результатов:")
    if task_type == 'regression':
        print("Type\t\tParameter\tVal Loss\tVal RMSE")
        for res in results:
            print(f"{res['type']}\t{res['param']}\t\t{res['val_loss']:.4f}\t\t{res['val_rmse']:.4f}")
    else:
        print("Type\t\tParameter\tVal Loss\tVal Acc\t\tVal F1")
        for res in results:
            print(
                f"{res['type']}\t{res['param']}\t\t{res['val_loss']:.4f}\t\t{res['val_acc']:.4f}\t\t{res['val_f1']:.4f}")


# Запускаем эксперименты для регрессии
print("\n" + "=" * 50)
print("ЭКСПЕРИМЕНТЫ ДЛЯ РЕГРЕССИИ (Auto MPG)")
print("=" * 50)
regression_results = run_experiments('regression')
visualize_results(regression_results, 'regression')

# Запускаем эксперименты для классификации
print("\n" + "=" * 50)
print("ЭКСПЕРИМЕНТЫ ДЛЯ КЛАССИФИКАЦИИ (Diabetes)")
print("=" * 50)
classification_results = run_experiments('classification')
visualize_results(classification_results, 'classification')
