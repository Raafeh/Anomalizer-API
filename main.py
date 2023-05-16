import numpy as np
import cv2
from keras.models import load_model
from collections import deque
from fastapi import FastAPI, File, UploadFile
from typing import List
import tensorflow as tf
import tempfile


app = FastAPI()

IMG_SIZE = 128  # Define the input image size

@app.post("/violence")
async def predict(video_file: UploadFile = File(...)):
    model = load_model('model.h5')
    Q = deque(maxlen=128)
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await video_file.read())
        video_path = tmp.name

    vs = cv2.VideoCapture(video_path)
    (W, H) = (None, None)
    count = 0
    result = []

    while True:
        (grabbed, frame) = vs.read()
        ID = vs.get(1)
        if not grabbed:
            break
        try:
            if ID % 7 == 0:
                count = count + 1
                n_frames = len(frame)

                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output = cv2.resize(frame, (512, 360)).copy()
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)).astype("float32")
                frame = frame.reshape(IMG_SIZE, IMG_SIZE, 3) / 255
                preds = model.predict(np.expand_dims(frame, axis=0))[0]
                Q.append(preds)

                results = np.array(Q).mean(axis=0)
                i = (preds > 0.56)[0]
                label = int(i)

                result.append(label)

                color = (0, 255, 0)
                if label:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)

                cv2.putText(output, f"Violence: {label}", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    vs.release()

    # Count instances of predictions
    count_violence = result.count(1)
    count_non_violence = result.count(0)

    # Determine the most frequent prediction
    if count_violence > count_non_violence:
        prediction = "violence"
    else:
        prediction = "non-violence"

    return {"prediction": prediction}

# Load the YOLOv4 TFLite model
model_path = "gun_fp16.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]



# Define the labels
labels = ['Gun']

@app.post("/gun_det")
async def detect_objects(video: UploadFile = File(...)):

    # Read video file
    video_bytes = await video.read()
    nparr = np.frombuffer(video_bytes, np.uint8)
    
    # Save video data to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
        temp_filename = temp_file.name
        temp_file.write(nparr)
        temp_file.flush()

        cap = cv2.VideoCapture(temp_filename)

    # Process video frames for object detection
    detections = []
    is_instance_detected = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, input_shape)
        input_data = np.expand_dims(resized_frame, axis=0)
        input_data = input_data.astype(np.float32)
        input_data /= 255.0

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run object detection
        interpreter.invoke()

        # Get the detection results
        output_data = []
        for output_detail in output_details:
            output_data.append(interpreter.get_tensor(output_detail['index']))

        # Process the results
        for i in range(len(output_data[0])):
            if np.any(output_data[0][i][2] >= 0.75):
                label = labels[int(output_data[1][i][0])]
                confidence = output_data[0][i][2]
                ymin, xmin, ymax, xmax = output_data[0][i][1]

                # Rescale the bounding box coordinates
                rescaled_xmin = int(xmin * frame.shape[1] / input_shape[1])
                rescaled_ymin = int(ymin * frame.shape[0] / input_shape[0])
                rescaled_xmax = int(xmax * frame.shape[1] / input_shape[1])
                rescaled_ymax = int(ymax * frame.shape[0] / input_shape[0])

                bbox = (rescaled_xmin, rescaled_ymin, rescaled_xmax, rescaled_ymax)
                detection = (label, bbox)
                detections.append(detection)
                is_instance_detected = True

        if np.any(is_instance_detected):
            break

    cap.release


    if len(detections) > 0:
        return detections
    else:
        return [("No objects detected", (0, 0, 0, 0))]

# Run the FastAPI application
# if __name__ == "_main_":
#     app.run()


# Load the YOLOv4 TFLite model
model_path1 = "yolov4-lug-fp16.tflite"
interpreter1 = tf.lite.Interpreter(model_path=model_path1)
interpreter1.allocate_tensors()

input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()
input_shape1 = input_details1[0]['shape'][1:3]



# Define the labels
labels1 = ['Suitcase','Backpack']

@app.post("/luggage_det")
async def detect_objects(video: UploadFile = File(...)):

    # Read video file
    video_bytes = await video.read()
    nparr = np.frombuffer(video_bytes, np.uint8)
    
    # Save video data to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
        temp_filename = temp_file.name
        temp_file.write(nparr)
        temp_file.flush()

        cap = cv2.VideoCapture(temp_filename)

    # Process video frames for object detection
    detections = []
    is_instance_detected = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, input_shape)
        input_data = np.expand_dims(resized_frame, axis=0)
        input_data = input_data.astype(np.float32)
        input_data /= 255.0

        # Set the input tensor
        interpreter1.set_tensor(input_details1[0]['index'], input_data)

        # Run object detection
        interpreter1.invoke()

        # Get the detection results
        output_data = []
        for output_detail in output_details1:
            output_data.append(interpreter1.get_tensor(output_detail['index']))

        # Process the results
        for i in range(len(output_data[0])):
            if np.any(output_data[0][i][2] >= 0.75):
                label = labels1[int(output_data[1][i][0][0])]
                confidence = output_data[0][i][2]
                ymin, xmin, ymax, xmax = output_data[0][i][1]

                # Rescale the bounding box coordinates
                rescaled_xmin = int(xmin * frame.shape[1] / input_shape[1])
                rescaled_ymin = int(ymin * frame.shape[0] / input_shape[0])
                rescaled_xmax = int(xmax * frame.shape[1] / input_shape[1])
                rescaled_ymax = int(ymax * frame.shape[0] / input_shape[0])

                bbox = (rescaled_xmin, rescaled_ymin, rescaled_xmax, rescaled_ymax)
                detection = (label, bbox)
                detections.append(detection)
                is_instance_detected = True

        if np.any(is_instance_detected):
            break

    cap.release

    if len(detections) > 0:
        return detections
    else:
        return [("No objects detected", (0, 0, 0, 0))]