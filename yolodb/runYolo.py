import cv2
import argparse
import random
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def calculate_calories_per_squat(weight_kg, height_cm):
    base_calories_per_squat = 0.225
    additional_calories_per_kg = 0.0044
    height_factor = 1 + (height_cm - 160) / 200
    
    return (base_calories_per_squat + additional_calories_per_kg * (weight_kg - 50)) * height_factor


ap = argparse.ArgumentParser()
# ap.add_argument('-m', type=str, required=True, help="Enter the .pt weight file path")
# ap.add_argument('-v', type=str, required=True, help="Enter the video file path")
ap.add_argument('--height', type=int, required=True, help="Enter your height")
ap.add_argument('--weight', type=int, required=True, help="Enter your weight")
args = vars(ap.parse_args())

# model = YOLO(args['m'])
model = YOLO('yolov8x-oiv7.pt')
names = model.names
# video_path = args['v']
video_path = 'c.mp4'

cap = cv2.VideoCapture(video_path)

assert cap.isOpened(), "Error reading video file."

db_cnt = 0
db_cnsp_factor = random.randint(70, 130) / 1000.0
consuption = calculate_calories_per_squat(args['height'], args['weight'])

frm_cnt = 0
pre_time = 0
pre_h = 0

lim_top = 50
lim_bot = 1000

while True:
    success, frame = cap.read()
    if not success:
        print("Fail to read the frame, please check the frame is empty or not.")
        break

    fps = round(1 / (time.time() - pre_time))
    pre_time = time.time()
    cv2.putText(frame, f"fps: {fps}", (500, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)

    results = model.predict(frame, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(frame, line_width=2, example=names)

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            if cls == 322:
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
                h = box[1]

                # if not pre_h == 0:
                if h > lim_top and pre_h < lim_top:
                    db_cnt += 1
                
                pre_h = h


    cv2.putText(frame, f"squat count: {db_cnt}", (30, 300), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, f"consumption: {round(consuption*db_cnt, 2)} kcol", (30, 330), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow("YOLO Test.", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frm_cnt += 1


cap.release()
cv2.destroyAllWindows()