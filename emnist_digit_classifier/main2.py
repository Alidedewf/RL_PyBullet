import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

# Проверка на наличие GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Трансформация изображений: нормализация + преобразование в тензор
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загрузка EMNIST (balanced)
dataset = datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=transform)

# Деление на 70% (train) и 30% (test)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Загрузчики данных
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Простейшая CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Меньше каналов
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Уменьшаем размер изображения в 2 раза

            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 47)  # 47 классов в EMNIST balanced
        )

    def forward(self, x):
        return self.net(x)

# Инициализация модели, функции потерь и оптимизатора
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Обратный проход + шаг оптимизации
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Тестирование после каждой эпохи
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Эпоха {epoch+1}/{epochs} | Потери: {running_loss/len(train_loader):.4f} | Точность на тесте: {accuracy:.2f}%")

# Сохранение модели после обучения
torch.save(model.state_dict(), "simplecnn.pth")
print("Модель сохранена в simplecnn.pth")