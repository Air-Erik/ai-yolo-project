from ultralytics import YOLO


def main():
    # YOLO; Объявление модели
    model = YOLO("yolov5n.pt")

    # YOLO; Обучение модели на наборе данных
    model.train(data='datasets/test_1v/data.yaml', epochs=100, imgsz=640, freeze=10)

    # YOLO; Проверка модели с помощью сета валидации
    model.val()


# Необходимое условие для многопоточности. Без данной конструкции не возможен параллелизм
if __name__ == '__main__':
    main()