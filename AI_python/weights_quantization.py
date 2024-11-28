import os
import numpy as np
import pandas as pd

# Папка, куда будут сохраняться квантованные веса
QUANTIZED_WEIGHTS_DIR = 'weights_quants'

# Создание папки для квантованных весов, если она не существует
if not os.path.exists(QUANTIZED_WEIGHTS_DIR):
    os.makedirs(QUANTIZED_WEIGHTS_DIR)

# Функция для квантования весов
def quantize_weights(weight_array, bit_depth=8):
    # Определяем диапазон для целых чисел (например, для 8 бит: от -128 до 127)
    min_val, max_val = -2**(bit_depth - 1), 2**(bit_depth - 1) - 1

    # Масштабируем веса в диапазон [min_val, max_val]
    scale_factor = np.max(np.abs(weight_array))
    if scale_factor == 0:
        return weight_array  # Если все значения равны нулю, не меняем

    # Масштабирование
    quantized_weights = np.round(weight_array / scale_factor * max_val)

    # Обрезаем значения, чтобы они попадали в допустимый диапазон
    quantized_weights = np.clip(quantized_weights, min_val, max_val)

    return quantized_weights

# Функция для обработки и квантования весов
def process_and_quantize_weights(input_dir='weights'):
    # Получаем все файлы в папке с весами
    weight_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for weight_file in weight_files:
        # Полный путь к файлу
        file_path = os.path.join(input_dir, weight_file)

        # Загружаем веса из CSV
        weight_data = pd.read_csv(file_path, header=None).values

        # Квантование весов
        quantized_data = quantize_weights(weight_data)

        # Сохранение квантованных весов в новый файл
        quantized_file_path = os.path.join(QUANTIZED_WEIGHTS_DIR, weight_file)
        pd.DataFrame(quantized_data).to_csv(quantized_file_path, index=False, header=False)

        print(f"Квантованные веса сохранены в: {quantized_file_path}")

# Вызов функции для квантования всех весов
process_and_quantize_weights()
