import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Загрузка и подготовка датасета CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Нормализация данных (приведение значений пикселей в диапазон [0, 1])
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Преобразование меток классов в формат one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Определение архитектуры классической сверточной нейронной сети
model = Sequential([
    # Сверточный слой 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    # Сверточный слой 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Сверточный слой 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Сверточный слой 4
    Conv2D(128, (3, 3), activation='relu', padding='same'),

    # Преобразование в плоский вектор
    Flatten(),

    # Полносвязный слой 1
    Dense(128, activation='relu'),

    # Полносвязный слой 2 (Sigmoid)
    Dense(64, activation='sigmoid'),

    # Полносвязный слой 3 (выходной)
    Dense(10, activation='softmax')

])


# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Вывод структуры модели
model.summary()

# Обучение модели
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точность на тестовых данных: {test_acc:.2f}")

# Сохранение всей модели в новом формате
model.save('base_AI_model.keras')
print("Модель сохранена в формате .keras")

# Сохранение только весов модели
model.save_weights('base_AI_weights.weights.h5')
print("Веса модели сохранены в файл base_AI_weights.weights.h5")

