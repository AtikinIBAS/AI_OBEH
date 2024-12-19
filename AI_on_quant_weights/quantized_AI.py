import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Функция для загрузки весов из файлов
def load_quantized_weights(model, weights_dir):
    for layer in model.layers:
        if len(layer.get_weights()) > 0:  # Если слой имеет веса
            weight_file = os.path.join(weights_dir, f"{layer.name}.csv")
            if os.path.exists(weight_file):
                # Загрузка весов из файла
                weights = np.loadtxt(weight_file, delimiter=",", dtype=np.float32)

                if len(layer.get_weights()) == 2:  # Если есть веса и смещения
                    bias_file = os.path.join(weights_dir, f"{layer.name}_biases.csv")
                    if os.path.exists(bias_file):
                        biases = np.loadtxt(bias_file, delimiter=",", dtype=np.float32)
                        layer.set_weights([weights.reshape(layer.get_weights()[0].shape), biases])
                    else:
                        layer.set_weights([weights.reshape(layer.get_weights()[0].shape)])
                else:
                    layer.set_weights([weights.reshape(layer.get_weights()[0].shape)])

# Загрузка и подготовка датасета CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Нормализация данных (приведение значений пикселей в диапазон [0, 1])
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Преобразование меток классов в формат one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Определение архитектуры нейросети
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3), name="conv2d_1"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2d_2"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same', name="conv2d_3"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same', name="conv2d_4"),

    Flatten(),
    Dropout(0.25),

    Dense(128, activation='sigmoid', name="dense_1"),
    Dropout(0.25),

    Dense(64, activation='sigmoid', name="dense_2"),
    Dropout(0.25),

    Dense(10, activation='softmax', name="dense_3"),
    Dropout(0.25),
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Применение квантизированных весов
weights_dir = "weights_quants"
load_quantized_weights(model, weights_dir)

# Обучение модели
model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точность на тестовых данных: {test_acc:.2f}")

"""
# Сохранение обученной модели в файл
save_path = "quantized_model.h5"
model.save(save_path)
print(f"Модель сохранена в файл: {save_path}")
"""

