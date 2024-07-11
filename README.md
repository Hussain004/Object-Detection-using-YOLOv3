<h1> Object Detection using YOLOv3 </h1>

<h2> Overview </h2>
This project utilizes the YOLOv3 (You Only Look Once) object detection algorithm to detect and identify various objects in real-time video streams. The project is implemented using Python and the OpenCV computer vision library.

<h2> Features </h2>
- Real-time object detection on live video streams
- Supports detection of 80 different object classes
- Displays bounding boxes and class labels around detected objects
- Utilizes the powerful YOLOv3 neural network model for efficient object detection

<h2> Required Files </h2>
The project consists of the following files:

1. Obj_Det.py: The main Python script that performs object detection and displays the results.
2. yolov3.cfg: The configuration file for the YOLOv3 neural network model.
3. coco.names: The list of 80 object classes that the YOLOv3 model can detect.

Note: The yolov3.weights file, which contains the pre-trained weights for the YOLOv3 model, is not included in this repository. You will need to download this file separately from the official YOLO website or other reliable sources and place it in the same directory as the other project files.

<h2> Installation and Setup </h2>
Install the required dependencies:
1. Python 3.x
2. OpenCV
3. Numpy
4. robomaster (optional, for integration with the Robomaster robot)
5. Download the yolov3.weights file from a reliable source and place it in the same directory as the other project files.
6. Run the Obj_Det.py script to start the object detection process.

<h2> How the Code Works </h2>
The Obj_Det.py script performs the following steps:

1. Load the pre-trained YOLOv3 model by reading the yolov3.weights and yolov3.cfg files.
2. Load the list of object classes from the coco.names file.
3. Initialize the Robomaster robot and its camera.
4. Define a detect_objects function that takes an input image and performs object detection using the YOLOv3 model.
5. The function creates a blob from the input image and passes it through the YOLOv3 network.
6. It then processes the output of the network to extract the bounding boxes, confidence scores, and class IDs of the detected objects.
7. The function applies non-maximum suppression to remove overlapping bounding boxes and returns the list of detected objects.
8. In the main loop, the script continuously reads frames from the Robomaster camera, detects objects in each frame, and displays the results with bounding boxes and class labels.
9. The program continues running until the user presses the 'q' key to exit.

<h2> Usage </h2>

1. Connect a camera or video source to your system.
2. Run the Obj_Det.py script.
3. The script will start detecting objects in the live video stream and display the results with bounding boxes and class labels.
4. To stop the program, press the 'q' key.

<h2> Credits </h2>
This project utilizes the YOLOv3 object detection algorithm, which was developed by Joseph Redmon and Ali Farhadi. The COCO dataset, which provides the 80 object classes used in this project, was created by the COCO Consortium.

<h2> Contribution </h2>
Contributions to this project are welcome. If you have any suggestions, bug reports, or improvements, please feel free to open an issue or submit a pull request on the project's GitHub repository.