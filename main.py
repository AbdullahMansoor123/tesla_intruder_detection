import random
import torch
import numpy as np
import cv2


def resize(image):
    return cv2.resize(image, (1080, 720))


model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.classes = [0]  # using person class only
model.conf = 0.3

zp_list = [(100, 100), (440, 100), (440, 380), (100, 380)]

# path = ""
v_cap = cv2.VideoCapture(0)
fps = int(v_cap.get(cv2.CAP_PROP_FPS))
frame_width = int(v_cap.get(3))
frame_height = int(v_cap.get(4))
num = random.randint(0, 100)
output = cv2.VideoWriter(f'output_yolov5_{num}.mp4', cv2.VideoWriter.fourcc(*'mp4v'), fps, (frame_width, frame_height))

# print(frame_width, frame_height)

person_in_roi = False
timer = 0
time_delay = 1.0  # Time threshold in seconds
while True:
    success, frame = v_cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (720, 640))
    frame = cv2.flip(frame, 1)
    cv2.polylines(frame, [np.array(zp_list, np.int32).reshape((-1, 1, 2))], True, (255, 0, 255), 2, cv2.LINE_AA)

    model_img = frame.copy()
    outputs = model(model_img)

    cv2.putText(frame, f"{time_delay:.2f}: {timer:.2f}", (80, 48), 1, 1, (0, 255, 0), 1, cv2.LINE_AA)
    person_insides = []
    for result in outputs.xyxy[0]:
        bbox = result[:4]
        conf = result[4]
        # row: coordinates of the detected person
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(frame, f"{conf:.2f}", (x1 + 5, y1 - 5), 1, 1, (0, 255, 255), 1, cv2.LINE_AA)
        bbox_center = int(x1 / 2) + int(x2 / 2), int(y1 / 2) + int(y2 / 2)
        cv2.circle(frame, bbox_center, 2, (0, 0, 255), 1)
        person_inside_roi = cv2.pointPolygonTest(np.array(zp_list), bbox_center, False)
        person_insides.append(person_inside_roi)

    print(person_insides)
    if any(class_id == 1.0 for class_id in person_insides):
        timer += 1 / fps
        # print('okay')
        if timer >= time_delay:
            print(f"wwwww^wwww^wwww^wwww^wwww^wwww^wwww^wwww^")
            output.write(frame)
    elif all(class_id == -1.0 for class_id in person_insides):
        timer = 0
        pass
    else:
        pass

    cv2.imshow("Output Video", resize(frame))
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
output.release()
v_cap.release()
cv2.destroyAllWindows()
