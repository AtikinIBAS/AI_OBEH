from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import csv

# Загрузка сохранённой модели
model = load_model("quantized_model.h5")

# Метки классов CIFAR-10
class_names = ['Самолёт', 'Автомобиль', 'Птица', 'Кошка', 'Олень', 'Собака', 'Лягушка', 'Лошадь', 'Корабль', 'Грузовик']

# Загрузка тестовых данных
(_, _), (x_test, y_test) = cifar10.load_data()

# Подготовка тестовых данных
x_test = x_test.astype('float32') / 255.0
y_test_onehot = to_categorical(y_test, 10)

# Предсказания для тестовых данных
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = y_test.flatten()

# Создание CSV файла с результатами
with open("mark_classes_q.csv", mode="w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Название метки", "Совпадение с результатом", "Предсказанная метка"])
    for i in range(len(true_labels)):
        true_class = class_names[true_labels[i]]
        pred_class = class_names[predicted_labels[i]]
        correct = 1 if true_labels[i] == predicted_labels[i] else 0
        writer.writerow([true_class, correct, pred_class])

print("Результаты сохранены в файл mark_classes_q.csv")
