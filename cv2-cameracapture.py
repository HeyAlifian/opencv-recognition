## PACKAGES ##
import tkinter as tk
import pyautogui
import numpy as np
import cv2
import colorama
import mediapipe as mp
from threading import Lock
from deepface import DeepFace

## FUNCTION ##
def check_duplicates(lst: list) -> list:
    origin_list = lst
    unique_list = []
    for item in origin_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

root = tk.Tk()
root.withdraw()

opencv_configs = {
    "videocapture-debugging": True,
    "videocapture-results-debugging": True,
    "speechrecognition-debugging": True,
    "videocapture-label": True
}

colorama.init(autoreset=True)
r = colorama.Fore.RED
g = colorama.Fore.GREEN
b = colorama.Fore.BLUE
yellow = colorama.Fore.YELLOW

prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
face_classifier = cv2.CascadeClassifier()
face_classifier.load(cv2.samples.findFile("models/haarcascade_frontalface_default.xml"))
min_confidence = 0.2

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

np.random.seed(123456789)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

global detected_objects, dominant_emotion, hand_positions
detected_objects = []
dominant_emotion = []
hand_positions = []
lock = Lock()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

mp_facemesh = mp.solutions.face_mesh
facemesh = mp_facemesh.FaceMesh(max_num_faces=1)

white_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(255, 255, 255))
red_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(0, 0, 255))

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.capture.set(cv2.CAP_PROP_FPS, 240)
        
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/120
        self.FPS_MS = int(self.FPS * 1000)
        
    def read_frame(self):
        if self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                return frame
            else:
                return None
        else:
            return None

threaded_camera = ThreadedCamera(0)
screen_width, screen_height = pyautogui.size()
x1 = y1 = x2 = y2 = 0

while True: 
    frame = threaded_camera.read_frame()
    frame = cv2.flip(frame, 1)
    if frame is None:
        continue
    (height, width) = frame.shape[:2]
    local_detected_objects = []
    local_dominant_emotion = []
    local_hand_positions = []
    try:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                idx = int(detections[0, 0, i, 1])
                label = classes[idx]
                local_detected_objects.append(label)
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                # Rectangle
                y = startY - 15 if startY - 15 > 15 else startY + 15
                if opencv_configs["videocapture-label"]:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                    cv2.putText(frame, label+f" {confidence:.2f}%", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    except Exception as e:
        print(f"[ERROR] Object Detection: {e}")
    if frame is not None:
        try:
            results = DeepFace.analyze(frame, actions=("emotion",), enforce_detection=False)
            if results:
                local_dominant_emotion.append(results[0]["dominant_emotion"])
                # Label
                if opencv_configs["videocapture-label"]:
                    cv2.putText(frame, f"{results[0]['dominant_emotion']}", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"[ERROR] Emotion Detection: {e}")

        try:
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(imgRGB)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    one_hand_landmarks = hand_landmarks.landmark
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                    y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                    # Determine the hand position relative to the screen's center
                    if x < width // 2:
                        position = "Hand on your left"
                    else:
                        position = "Hand on your right"
                    local_hand_positions.append(position)

                for hand in result.multi_hand_landmarks:
                    one_hand_landmarks = hand_landmarks.landmark
                    for id, lm in enumerate(one_hand_landmarks):
                        x = int(lm.x * width)
                        y = int(lm.y * height)
                        match id:
                            case 8:
                                mouse_x: int = int(screen_width / width * x)
                                mouse_y: int = int(screen_height / height * y)
                                cv2.circle(frame, (x, y), 25, (255,255,255))
                                x1 = x
                                y1 = y
                            case 4:
                                x2 = x
                                y2 = y
                                cv2.circle(frame, (x, y), 25, (255,255,255))
                            case _:
                                ...
                    #pyautogui.moveTo(mouse_x, mouse_y)
                dist: any = y2 - y1
        except Exception as e:
            print(f"[ERROR] Hand Tracking: {e}")
        try:
            facemesh_results = facemesh.process(imgRGB)
            if facemesh_results.multi_face_landmarks:
                for face_landmarks in facemesh_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_facemesh.FACEMESH_CONTOURS, landmark_drawing_spec=red_spec, connection_drawing_spec=white_spec)
        except Exception as e:
            print(f"[ERROR] Face Mesh: {e}")
    with lock:
        detected_objects = check_duplicates(local_detected_objects)
        dominant_emotion = check_duplicates(local_dominant_emotion)
        hand_positions = check_duplicates(local_hand_positions)

    if opencv_configs["videocapture-results-debugging"]:
        print(f"""
OPENCV RESULT
-------------------------
DETECTED OBJECTS: {detected_objects}
DOMINANT EMOTION: {dominant_emotion}
HAND POSITION: {hand_positions}
""")
    if opencv_configs["videocapture-debugging"]:
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
threaded_camera.capture.release()
cv2.destroyAllWindows()