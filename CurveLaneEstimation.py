import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
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

def plot_show(x, y):
    # plt.ion()
    plt.plot(x, y)
    plt.show()
    # plt.pause(0.1)
    # plt.clf()

def run_ransac(lane, side):
    # Get the x and y coordinates of the lane pixels
    x = np.array([])
    y = np.array([])
    for i in range(lane.shape[0]):
        if side == 'right':
            for j in range(lane.shape[1]):
                if lane[i, j] > 0:
                    x = np.append(x, j)
                    y = np.append(y, lane.shape[0] - i)
                    break
        else:
            for j in range(lane.shape[1] - 1, -1, -1):
                if lane[i, j] > 0:
                    x = np.append(x, j)
                    y = np.append(y, lane.shape[0] - i)
                    break
    
    x_min = int(np.min(x))
    x_max = int(np.max(x))

    runsac_iterations = 500
    best_spline = (None, None)
    for i in range(runsac_iterations):
        # Randomly sample 4 points
        sample_indices = np.random.choice(len(x), 4, replace=False)
        sample_x = x[sample_indices]
        sample_y = y[sample_indices]

        # Fit a cubic spline to the sampled points
        spline = np.zeros((4, 1)) 
        feature_mat = np.array([sample_x**3, sample_x**2, sample_x, np.ones_like(sample_x)]).T
        output_mat = sample_y.reshape(-1, 1)

        spline = np.linalg.pinv(feature_mat.T @ feature_mat) @ feature_mat.T @ output_mat

        # Evaluate the spline on all x values
        y_pred = spline[0] * x**3 + spline[1] * x**2 + spline[2] * x + spline[3]

        # Get the inliers coordinates(x, y) based on the threshold
        inliers = None
        threshold = 1

        for j in range(len(x)):
            if abs(y_pred[j] - y[j]) < threshold:
                if inliers is None:
                    inliers = (x[j], y[j])
                else:
                    inliers = np.vstack((inliers, (x[j], y[j])))

        # Update the best spline if the current spline has more inliers
        if best_spline[1] is None or (inliers is not None and len(inliers) > len(best_spline[1])):
            best_spline = (spline, inliers)

    # Fit the final spline using all inliers
    # feature_mat = np.array([best_spline[1][:, 0]**3, best_spline[1][:, 0]**2, best_spline[1][:, 0], np.ones_like(best_spline[1][:, 0])]).T
    # output_mat = best_spline[1][:, 1].reshape(-1, 1)

    best_spline = best_spline[0].flatten()

    interval = np.linspace(x_min, x_max, 100)
    y_pred = best_spline[0] * interval**3 + best_spline[1] * interval**2 + best_spline[2] * interval + best_spline[3]

    image_y = (lane.shape[0] - y_pred).astype(int)

    points = np.column_stack((interval.astype(int), image_y)).reshape(-1, 1, 2)

    return spline, points

def get_ROI_lanemarks(edges):
    # Get the ROI for the left and right lanes
    lane = edges[2 * edges.shape[0] // 5:, :]
    left_lane = lane[:, :edges.shape[1] // 3]
    right_lane = lane[:, 4 * edges.shape[1] // 5:]

    return left_lane, right_lane

def get_road_segment(left_lane, right_lane, spline_left, spline_right, len_sq_road):
    # Choose some point of the left lane
    x_left, y_left = left_lane[left_lane.shape[0] // 2 - 120, 0]

    # Search in the right_lane points for the point that minimizez the distance with the len_sq_road
    x_right_best, y_right_best = 0, 0
    best_error = int(1e9)
    for i in range(right_lane.shape[0]):
        x_right, y_right = right_lane[i, 0]
        if np.abs((x_left - x_right)**2 + (y_left - y_right)**2 - len_sq_road) < best_error:
            x_right_best = x_right
            y_right_best = y_right
            best_error = np.abs((x_left - x_right)**2 + (y_left - y_right)**2 - len_sq_road)

    print(f"Best error: {best_error}")
    return np.arctan2((y_right_best - y_left), (x_right_best - x_left)), np.array([[x_left, y_left], [x_right_best, y_right_best]])

def lane_detecton_pipeline(image):
    # Convert to grayscale and apply Canny edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Find the edges using Canny
    edges = cv2.Canny(gray_image, 100, 150)
    # edges = cv2.erode(edges, (3, 3), iterations=1)
    # edges = cv2.dilate(edges, (3, 3), iterations=1)

    # Get the ROI for the left and right lanes
    left_lane, right_lane = get_ROI_lanemarks(edges)

    # Run RUNSAC for best spline fit for left and right lanes
    spline_left, points_left = run_ransac(lane=left_lane, side='left')
    spline_right, points_right = run_ransac(lane=right_lane, side='right')

    # Add the offset to the y coordinates
    points_left = points_left + np.array([0, 2 * image.shape[0] // 5])
    points_right = points_right + np.array([4 * image.shape[1] // 5, 2 * image.shape[0] // 5])

    # Draw the lane lines on the image
    image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    image = cv2.polylines(image, [points_left], isClosed=False, color=(0, 255, 0), thickness=10)
    image = cv2.polylines(image, [points_right], isClosed=False, color=(0, 255, 0), thickness=10)

    # Add the line to describe the front of the car
    line_front_car = get_car_front_line(image)
    image = cv2.polylines(image, [line_front_car], isClosed=False, color=(255, 0, 0), thickness=10)

    # Add the line to describe curvature of the road
    angle, line_road = get_road_segment(points_left, points_right, spline_left, spline_right, (1000)**2)
    image = cv2.polylines(image, [line_road], isClosed=False, color=(0, 0, 255), thickness=10)

    # Add to image the text that it need to steer in degrees
    cv2.putText(image, f"Steer: {angle:.2f} degrees", (image.shape[1] // 2, image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image

# Function to estimate distance based on bounding box size
def estimate_distance(bbox_width, bbox_height):
    # For simplicity, assume the distance is inversely proportional to the box size
    # This is a basic estimation, you may use camera calibration for more accuracy
    focal_length = 1000  # Example focal length, modify based on camera setup
    known_width = 2.0  # Approximate width of the car (in meters)
    distance = (known_width * focal_length) / bbox_width  # Basic distance estimation
    return distance

def get_car_front_line(image):
    # Return horizontal line representing the front of the car
    return np.array([[image.shape[1] // 4, 4 * image.shape[0] // 5],
                     [image.shape[1] - image.shape[1] // 4, 4 * image.shape[0] // 5]])

model = YOLO('yolov8n.pt')

video = cv2.VideoCapture("input.mp4")

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
        cv2.imshow('Lane Detection', lane_frame)
        cv2.waitKey(1)