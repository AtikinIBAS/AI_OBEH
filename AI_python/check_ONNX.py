import tf2onnx
import onnx

# Преобразование модели SavedModel в формат ONNX
onnx_model, _ = tf2onnx.convert.from_saved_model("saved_model_base_AI", opset=13)

# Сохранение модели в формате ONNX
onnx.save_model(onnx_model, "base_AI_model.onnx")
print("Модель успешно преобразована в ONNX и сохранена как base_AI_model.onnx")
