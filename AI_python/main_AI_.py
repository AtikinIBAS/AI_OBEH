import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Загрузка и подготовка датасета CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Нормализация данных (приведение значений пикселей в диапазон [0, 1])
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Преобразование меток классов в формат one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Определение архитектуры модели
model = Sequential([
    # Сверточный слой 1
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    # Сверточный слой 2
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Сверточный слой 3
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Сверточный слой 4
    Conv2D(128, (3, 3), activation='relu', padding='same'),

    # Преобразование в плоский вектор
    Flatten(),

    # Полносвязный слой 1
    Dense(64, activation='relu'),

    # Полносвязный слой 2
    Dense(32, activation='sigmoid'),

    # Полносвязный слой 3 (выходной)
    Dense(10, activation='softmax')
])

# Компиляция модели с оптимизатором SGD
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Вывод структуры модели
model.summary()

# Обучение модели
history = model.fit(x_train, y_train, epochs=15, batch_size=128, validation_split=0.2)

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точность на тестовых данных: {test_acc:.2f}")

