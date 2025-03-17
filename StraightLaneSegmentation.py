import numpy as np
from ultralytics import YOLO
import math
import cv2

# Function to draw the filled polygon between the lane lines
def draw_lane_lines(img, left_line, right_line, color=[0, 255, 0], thickness=10):
    line_img = np.zeros_like(img)
    poly_pts = np.array([[
        (left_line[0], left_line[1]),
        (left_line[2], left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1])
    ]], dtype=np.int32)
    
    # Fill the polygon between the lines
    cv2.fillPoly(line_img, poly_pts, color)
    
    # Overlay the polygon onto the original image
    img = cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0)
    return img

def lane_detecton_pipeline(image):
    # Convert to grayscale and apply Canny edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray_image, 100, 150)
    # Perform Hough Line Transformation to detect lines

    lines = cv2.HoughLinesP(
        edges,
        rho=10,
        theta=np.pi / 180,
        threshold=200,
        lines=np.array([]),
        minLineLength=150,
        maxLineGap=50
    )

    # Separating left and right lines based on slope
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if lines is None:
        print("No lines detected")
        return image

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if math.fabs(slope) < 0.5:  # Ignore nearly horizontal lines
                continue
            if slope <= 0:  # Left lane
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:  # Right lane
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    # Fit a linear polynomial to the left and right lines
    min_y = int(image.shape[0] * (1 / 5))  # Slightly below the middle of the image
    max_y = image.shape[0]  # Bottom of the image

    if left_line_x and left_line_y:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
    else:
        left_x_start, left_x_end = 0, 0  # Defaults if no lines detected

    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
    else:
        right_x_start, right_x_end = 0, 0  # Defaults if no lines detected

    # Create the filled polygon between the left and right lane lines
    lane_image = draw_lane_lines(
        image,
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y]
    )

    return lane_image


# Function to estimate distance based on bounding box size
def estimate_distance(bbox_width, bbox_height):
    # For simplicity, assume the distance is inversely proportional to the box size
    # This is a basic estimation, you may use camera calibration for more accuracy
    focal_length = 1000  # Example focal length, modify based on camera setup
    known_width = 2.0  # Approximate width of the car (in meters)
    distance = (known_width * focal_length) / bbox_width  # Basic distance estimation
    return distance

model = YOLO('yolov8n.pt')

video = cv2.VideoCapture("test.mp4")

while True:
    success, frame = video.read()

    if not success:
        break
    
    # Resize frame to 720p
    frame = cv2.resize(frame, (1280, 720))

    # Run the lane detection pipeline
    lane_frame = lane_detecton_pipeline(frame)

    # Run the model on the frame
    results = model.predict(frame)

    result = results[0]

    # Process the detections
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class ID

        # Only draw bounding boxes for cars with confidence >= 0.5
        if model.names[cls] == 'car' and conf >= 0.5:
            label = f'{model.names[cls]} {conf:.2f}'

            # Draw the bounding box
            cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(lane_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Estimate the distance of the car
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            distance = estimate_distance(bbox_width, bbox_height)

            # Display the estimated distance
            distance_label = f'Distance: {distance:.2f}m'
            cv2.putText(lane_frame, distance_label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the resulting frame with both lane detection and car detection
        cv2.imshow('frame', lane_frame)
        cv2.waitKey(1)