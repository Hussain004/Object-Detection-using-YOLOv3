import cv2
import numpy as np
from robomaster import robot

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


ep_robot = robot.Robot()
ep_robot.initialize(conn_type="ap")

ep_camera = ep_robot.camera
ep_chassis = ep_robot.chassis
ep_camera.start_video_stream(display=False)



def detect_objects(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            detected_objects.append((x, y, w, h, class_ids[i]))
    return detected_objects



while True:
    img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
    detected_objects = detect_objects(img)
    
    for (x, y, w, h, class_id) in detected_objects:
        label = str(classes[class_id])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

ep_camera.stop_video_stream()
ep_robot.close()
cv2.destroyAllWindows()
