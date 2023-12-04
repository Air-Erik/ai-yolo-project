from ultralytics import YOLO
#import clearml

#clearml.browser_login()

# Объявление модели
model = YOLO("yolov5m.pt")

# Обучение модели на наборе данных
result = model.train(data='datasets/test_1v/data.yaml', epochs=100, imgsz=640)

# Оценка качества тренировки модели на проверочном сете
result = model.val()