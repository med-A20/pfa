import mediapipe as mp
import numpy as np
import cv2

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.vision.ObjectDetectorResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a VideoCapture object to access the webcam (usually camera index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Set the width and height of the display window
window_width = 800
window_height = 600

cap.set(cv2.CAP_PROP_XI_WIDTH, 800)
cap.set(cv2.CAP_PROP_XI_HEIGHT, 600)

def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    print('detection result: {}'.format(result))
    for detection in result.detections:
        for category in detection.categories:
            if category.score > 0.65:
                box = detection.bounding_box
                # Draw a rectangle on the frame
                cv2.rectangle(frame, (box.origin_x, box.origin_y), (box.origin_x + box.width, box.origin_y + box.height),
                              (0, 255, 0), 2)
                # Add text in the top-left corner
                text = f"{category.category_name} {category.score}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (255, 255, 255)  # White color
                thickness = 2
                text_position = (box.origin_x, box.origin_y)  # Coordinates for top-left corner
                # Draw lines from bottom right and bottom left to the center
                # Calculate the center of the rectangle
                center_x = window_width // 2
                center_y = window_height // 2
                bottom_right = (0, window_height)
                bottom_left = (window_width, window_height)
                cv2.line(frame, bottom_right, (center_x, center_y), (255, 0, 0), 2)  # Blue line
                cv2.line(frame, bottom_left, (center_x, center_y), (0, 0, 255), 2)  # Red line
                cv2.putText(frame, text, text_position, font, font_scale, font_color, thickness)




options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='./efficientdet_lite0.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result)

# Loop to continuously capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    with ObjectDetector.create_from_options(options) as detector:
        # Send the latest frame to perform object detection.
        # Results are sent to the `result_callback` provided in the `ObjectDetectorOptions`.
        detector.detect_async(mp_image, 50000)


    # Display the captured frame in a window
    cv2.imshow("Webcam Feed", frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the window
cap.release()
cv2.destroyAllWindows()

