import csv
import matplotlib.pyplot as plt

# Функция для чтения данных из CSV
def read_from_csv(file_path):
    data = []
    epoch_data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Пропуск заголовка
        for row in reader:
            if row[0].startswith("Epoch"):
                epoch_data.append((row[0], float(row[1])))
            else:
                data.append((row[0], float(row[1])))
    return data, epoch_data

# Построение графика времени по этапам
def plot_stage_timing(data):
    stages = [stage for stage, _ in data]
    times = [time for _, time in data]

    plt.figure(figsize=(10, 6))
    plt.bar(stages, times, color='skyblue')
    plt.xlabel('Этапы')
    plt.ylabel('Время (в секундах)')
    plt.title('Время выполнения по этапам')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Построение графика времени обучения по эпохам
def plot_epoch_timing(epoch_data):
    epochs = [epoch for epoch, _ in epoch_data]
    times = [time for _, time in epoch_data]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, times, marker='o', color='orange', label='Время на эпоху')
    plt.xlabel('Эпохи')
    plt.ylabel('Время (в секундах)')
    plt.title('Время обучения по эпохам')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Путь к файлу CSV
csv_file = "timing_data_q.csv"
timing_data, epoch_timing = read_from_csv(csv_file)

# Построение графиков
plot_stage_timing(timing_data)
plot_epoch_timing(epoch_timing)
