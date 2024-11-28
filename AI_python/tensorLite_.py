import tensorflow as tf

# Загрузка обученной модели
model = tf.keras.models.load_model('base_AI_model.keras')

# Преобразование модели в формат TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Сохранение TFLite-модели
with open('base_AI_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Модель преобразована в TensorFlow Lite.")
