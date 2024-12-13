import os
import tensorflow as tf
import pandas as pd
import psutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

# Загрузка и подготовка датасета CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Определение архитектуры нейросети
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),

    Flatten(),
    Dropout(0.25),

    Dense(128, activation='sigmoid'),
    Dropout(0.25),

    Dense(64, activation='sigmoid'),
    Dropout(0.25),

    Dense(10, activation='softmax'),
    Dropout(0.25),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Создаем колбэк для замера ресурсов
class ResourceMonitor(Callback):
    def on_train_begin(self, logs=None):
        self.cpu_usage = []
        self.memory_usage = []
        self.process = psutil.Process(os.getpid())

    def on_epoch_end(self, epoch, logs=None):
        # Получаем текущую загрузку CPU
        cpu_percent = psutil.cpu_percent(interval=None)  # загрузка CPU в %

        # Получаем информацию о памяти, занятой текущим процессом
        mem_info = self.process.memory_info()
        mem_used = mem_info.rss / (1024**3)  # используемая память в ГБ для текущего процесса

        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(mem_used)

# Инициализируем колбэк
resource_monitor = ResourceMonitor()

# Обучение модели с колбэком
model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2, callbacks=[resource_monitor])

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точность на тестовых данных: {test_acc:.2f}")

# Сохранение показаний CPU и памяти в CSV
df = pd.DataFrame({
    "Epoch": list(range(1, len(resource_monitor.cpu_usage) + 1)),
    "CPU_Usage(%)": resource_monitor.cpu_usage,
    "Memory_Used(GB)": resource_monitor.memory_usage
})

df.to_csv("metrics.csv", index=False)
print("Показатели загрузки ресурсов сохранены в metrics.csv")
