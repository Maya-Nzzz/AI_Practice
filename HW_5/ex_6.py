import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from datasets import CustomImageDataset
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Функция для визуализации
def plot_training_history(
        history: Dict[str, List[float]],
        num_epochs: int,
        save_path: Optional[str],
        title: Optional[str]
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = [x for x in range(1, num_epochs+1)]
    # Loss plot
    ax1.plot(epochs, history['train_losses'], marker='o', linestyle='-', label='Train Loss')
    ax1.plot(epochs, history['test_losses'], marker='o', linestyle='-', label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history['train_accs'], marker='o', linestyle='-', label='Train Acc')
    ax2.plot(epochs, history['test_accs'], marker='o', linestyle='-', label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

# Подготовка датасета
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = CustomImageDataset('data/train', transform=transform)
val_dataset = CustomImageDataset('data/test', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Загрузка модели
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(train_dataset.get_class_names()))

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# История
history = {
    'train_losses': [],
    'test_losses': [],
    'train_accs': [],
    'test_accs': []
}

num_epochs = 3

# Цикл обучения
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, predicted = torch.max(out, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Валидация
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            val_loss += loss.item() * x.size(0)
            _, predicted = torch.max(out, 1)
            val_correct += (predicted == y).sum().item()
            val_total += y.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    history['train_losses'].append(train_loss)
    history['test_losses'].append(val_loss)
    history['train_accs'].append(train_acc)
    history['test_accs'].append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

plot_training_history(history, num_epochs, save_path='results/ex_6.png', title='EfficientNetB0 Training History')
print("Графики сохранены в training_history.png")
