import tensorflow as tf
import tf2onnx
import onnx

# Загрузка обученной модели
model = tf.keras.models.load_model('base_AI_model.keras')

# Сохранение модели в формате SavedModel
model.export("saved_model_base_AI")
print("Модель сохранена в формате SavedModel.")

# Преобразование SavedModel в ONNX
onnx_model, _ = tf2onnx.convert.from_saved_model("saved_model_base_AI", opset=13)

# Сохранение ONNX модели
onnx.save_model(onnx_model, 'base_AI_model.onnx')
print("Модель успешно преобразована в ONNX и сохранена как base_AI_model.onnx")
