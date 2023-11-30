from ultralytics import YOLO

# Объявление модели
model = YOLO("yolov5s.yaml")

# Обучение модели на наборе данных
result = model.train(data='datasets/test_1v/data.yaml', epochs=30, imgsz=640)

# Оценка качества тренировки модели на проверочном сете
result = model.val()