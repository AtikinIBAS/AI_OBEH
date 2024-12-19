import time
import os
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Функция для записи данных в .csv
def write_to_csv(file_path, data):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Stage", "Time (seconds)"])
        writer.writerows(data)

# Загрузка и подготовка датасета CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Начало замера времени
timing_data = []
epoch_timing = []

start_time = time.time()

# Нормализация данных (приведение значений пикселей в диапазон [0, 1])
stage_start = time.time()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
timing_data.append(["Normalization", time.time() - stage_start])

# Преобразование меток классов в формат one-hot encoding
stage_start = time.time()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
timing_data.append(["One-hot Encoding", time.time() - stage_start])

# Определение архитектуры нейросети
stage_start = time.time()
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
timing_data.append(["Model Definition", time.time() - stage_start])

# Компиляция модели
stage_start = time.time()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
timing_data.append(["Model Compilation", time.time() - stage_start])

# Обучение модели
num_epochs = 15
for epoch in range(num_epochs):
    stage_start = time.time()
    model.fit(x_train, y_train, epochs=1, batch_size=64, validation_split=0.2, verbose=1)
    epoch_timing.append([f"Epoch {epoch + 1}", time.time() - stage_start])

timing_data.append(["Model Training", sum([time for _, time in epoch_timing])])

# Конец замера времени
timing_data.append(["Total Time", time.time() - start_time])

# Сохранение данных времени в CSV
csv_file = "timing_data.csv"
write_to_csv(csv_file, timing_data + epoch_timing)
print(f"Данные времени сохранены в {csv_file}")

# Оценка модели на тестовых данных
stage_start = time.time()
test_loss, test_acc = model.evaluate(x_test, y_test)
timing_data.append(["Model Evaluation", time.time() - stage_start])
