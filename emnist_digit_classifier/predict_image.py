import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

# === Архитектура CNN как в твоей модели ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 47)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# === Классы EMNIST (balanced) ===
emnist_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# === Трансформация, как в EMNIST ===
transform = transforms.Compose([
    transforms.Grayscale(),               # ч/б
    transforms.Resize((28, 28)),          # 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === EMNIST поворот (обязательно!) ===
def emnist_fix(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)

# === Предсказание символа по пути к изображению ===
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Файл не найден: {image_path}")
        return

    image = Image.open(image_path).convert("L")
    image = emnist_fix(image)              # Поворот, как требует EMNIST
    image = transform(image).unsqueeze(0)  # [1, 1, 28, 28]

    model = SimpleCNN()
    try:
        model.load_state_dict(torch.load("simplecnn.pth", map_location="cpu"))
    except Exception as e:
        print("❌ Ошибка при загрузке модели:", e)
        return

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_char = emnist_classes[predicted.item()]
        print(f"✅ {image_path} → Распознанный символ: {predicted_char}")

# === Точка входа ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🔹 Использование: python predict_image.py <image_path>")
    else:
        predict_image(sys.argv[1])