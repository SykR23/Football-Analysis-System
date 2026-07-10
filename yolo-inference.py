import ultralytics
from ultralytics import YOLO
model = YOLO('model/best.pt')
result = model('Input_videos/08fd33_4.mp4', save=True)
for box in result[0].boxes:
    print(box)