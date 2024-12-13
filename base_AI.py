import tensorflow as tf
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Колбэк для сохранения весов
class WeightsSaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def on_epoch_end(self, epoch, logs=None):
        for i, layer in enumerate(self.model.layers):
            weights = layer.get_weights()
            if weights:  # Если у слоя есть веса
                for j, weight in enumerate(weights):
                    layer_name = layer.name
                    weight_filename = f'{self.save_dir}/{layer_name}_weights_{i + 1}_weights_{j + 1}.csv'

                    # Проверяем, если веса имеют 4D форму (для Conv2D)
                    if weight.ndim == 4:  # Веса для свёрточных слоёв
                        # Преобразуем их в двумерный массив (перемешаем все элементы в одномерный массив)
                        flattened_weights = weight.flatten()
                        # Сохраняем их как одномерный CSV
                        pd.DataFrame(flattened_weights).to_csv(weight_filename, index=False)
                    else:
                        # Для остальных слоёв сохраняем как есть (например, для Dense слоёв)
                        pd.DataFrame(weight).to_csv(weight_filename, index=False)
                    print(f"Сохранены веса слоя {layer_name} в {weight_filename}")

# Загрузка и подготовка данных
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Определение архитектуры нейросети
# Определение архитектуры CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Снижение переобучения

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='sigmoid'),
    Dropout(0.5),
    Dense(64, activation='sigmoid'),
    Dense(10, activation='softmax')  # Выходной слой
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Вставляем наш колбэк для сохранения весов
weights_saver = WeightsSaverCallback(save_dir='weights')

# Обучение модели
model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2, callbacks=[weights_saver])

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точность на тестовых данных: {test_acc:.2f}")

