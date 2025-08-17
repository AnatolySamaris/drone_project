import os
from ament_index_python.packages import get_package_share_directory
import numpy as np
from torch import nn
import torch

from collections import deque
import matplotlib
matplotlib.use('agg')  # Отключаем интерактивный режим для скорости
import matplotlib.pyplot as plt
import cv2


class ValueFilter:
    """
    Для фильтрации управляющего воздействия.
    Применяется скользящее среднее и экспоненциальное сглаживание
    для снижения влияния шумов.
    """
    def __init__(self, window_size: int, alpha=0.2) -> None:
        self.window = deque([], maxlen=window_size)
        self.alpha = alpha
        self.smoothed_ema = None
    
    def update(self, new_value) -> float:
        self.window.append(new_value)

        # Медиана, для отсечения выбросов
        median = np.median(self.window)

        # Скользящее среднее
        cleaned = [x for x in self.window if abs(x - median) < 2 * np.std(self.window)]
        ma = np.mean(cleaned) if cleaned else median  # Если все значения сильно отличаются, берём медиану

        # Экспоненциальное сглаживание
        if self.smoothed_ema is None:
            self.smoothed_ema = ma
        else:
            self.smoothed_ema = self.alpha * ma + (1 - self.alpha) * self.smoothed_ema
        
        return self.smoothed_ema
    

class PlotWatcher:
    def __init__(self, values_titles: list[str], window_size: int, figsize=(10, 6), plot_title=None):
        self.plot_title = plot_title
        self.figsize = figsize
        self.values_titles = values_titles
        self.num_of_values = len(values_titles)
        self.window_size = window_size
        self.value_windows = [deque([], maxlen=window_size) for _ in range(self.num_of_values)]
        self.t_range = range(self.window_size)
    
    def update(self, *new_values) -> np.ndarray:
        # Добавление новых значений
        for i, new_value in enumerate(new_values):
            self.value_windows[i].append(new_value)

        # Рисование графика
        plt.figure(figsize=self.figsize, dpi=100)
        plt.title(self.plot_title, fontsize=14)
        plt.xlabel('Время, мс', fontsize=12)
        plt.ylabel('Значение', fontsize=12)
        plt.xticks([])  # Скрыть все метки оси X
        for i, window in enumerate(self.value_windows):
            if i % 2 == 0:
                plt.plot(self.t_range[:len(window)], window, linewidth=2, label=self.values_titles[i], linestyle='--')
            else:
                plt.plot(self.t_range[:len(window)], window, linewidth=2, label=self.values_titles[i])
        plt.legend()
        plt.grid(True, linestyle='--')

        # Перевод графика в формат картинки
        fig = plt.gcf()
        fig.canvas.draw()
        img_rgb = np.array(fig.canvas.buffer_rgba())    # H, W, 4
        img_bgr = cv2.cvtColor(img_rgb[..., :3], cv2.COLOR_RGB2BGR)

        # Закрываем график для экономии памяти
        matplotlib.pyplot.close()
        return img_bgr

# ==============================================
# ==============================================
# ==============================================

class GestureClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 168)
        self.fc2 = nn.Linear(168, 546)
        self.fc3 = nn.Linear(546, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def load_classifier_model(input_size=42, num_classes=8):
    package_dir = get_package_share_directory("cv_control")
    model_path = os.path.join(package_dir, "models", "gesture_classifier.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
       
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = GestureClassifier(input_size, num_classes)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device(device)
        )
    )
    model.to(device)
    model.eval()
    return model
