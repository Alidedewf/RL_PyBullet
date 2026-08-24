# emnist_digit_classifier

CNN на PyTorch для распознавания рукописных символов датасета [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset).

| Файл | Роль |
|---|---|
| `main.py` | Базовая версия обучения CNN |
| `main2.py` / `main3.py` / `main4.py` | Итерации — эксперименты с архитектурой/гиперпараметрами поверх `main.py` |
| `predict_image.py` | Загружает `simplecnn.pth` и распознаёт произвольное изображение (см. `A_sample*.png`) |
| `simplecnn.pth` | Обученные веса (413 КБ) |

```bash
pip install torch torchvision pillow numpy
python predict_image.py A_sample.png
```

Датасет EMNIST не включён в репозиторий — скачивается автоматически через `torchvision.datasets.EMNIST` при первом запуске `main.py`.
