import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из CSV-файла
df = pd.read_csv("mark_classes_q.csv")

# Группировка по названию метки и подсчет суммы верных предсказаний
correct_counts = df.groupby("Название метки")["Совпадение с результатом"].sum()

# Построение графика
plt.figure(figsize=(10, 6))
correct_counts.plot(kind='bar', color='green', alpha=0.7)

plt.title("Количество верно угаданных значений для каждой метки класса")
plt.xlabel("Метка класса")
plt.ylabel("Количество верных предсказаний")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()
