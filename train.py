from ultralytics import YOLO

def main():
    # YOLO; Объявление модели
    model = YOLO("yolov5n.pt")
    
    # YOLO; Обучение модели на наборе данных
    result = model.train(data='datasets/test_1v/data.yaml', epochs=100, imgsz=640, freeze=10)

# Необходимое условие для многопоточности. Без данной конструкции не возможен параллелизм
if __name__ == '__main__':
    main()