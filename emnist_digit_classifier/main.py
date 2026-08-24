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
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 1 → 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 → 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16 → 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 → 7x7

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7 → 3x3
        )

        self.fc = nn.Sequential(
            nn.Flatten(),  # 64 * 3 * 3 = 576
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 47)  # EMNIST имеет 47 классов
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
# Инициализация модели, функции потерь и оптимизатора
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
epochs = 30
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