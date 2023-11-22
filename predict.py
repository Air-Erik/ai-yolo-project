from ultralytics import YOLO
from PIL import Image
import cv2

# Загрузка модели для пердсказания
model = YOLO("/runs/detect/train/weights/best.pt")