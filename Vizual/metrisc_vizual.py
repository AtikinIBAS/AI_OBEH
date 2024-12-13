import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из CSV-файла
df = pd.read_csv("metrics.csv")

# Извлекаем данные
epochs = df["Epoch"]
cpu_usage = df["CPU_Usage(%)"]
memory_used = df["Memory_Used(GB)"]

# Создаём окно с подграфиками
plt.figure(figsize=(10, 6))

# График загрузки CPU по эпохам
plt.subplot(2, 1, 1)
plt.plot(epochs, cpu_usage, marker='o', color='blue', label='CPU Usage')
plt.title("Использование CPU по эпохам")
plt.xlabel("Эпоха")
plt.ylabel("CPU Usage (%)")
plt.grid(True)
plt.legend()

# График использования памяти по эпохам
plt.subplot(2, 1, 2)
plt.plot(epochs, memory_used, marker='o', color='green', label='Memory Used')
plt.title("Использование памяти по эпохам")
plt.xlabel("Эпоха")
plt.ylabel("Использование памяти (GB)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
