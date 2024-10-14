# %%
import cv2
import numpy as np
import math
import argparse
import time

# Set up argument parser for command-line flags
parser = argparse.ArgumentParser(description="Clock Detection from Video, Image, or Webcam.")
parser.add_argument("--mode", type=str, choices=["video", "image", "webcam"], required=True,
                    help="Select input mode: 'video', 'image', or 'webcam'.")
parser.add_argument("--path", type=str, help="Path to the video or image file.")
args = parser.parse_args()

# Initialize video capture based on the selected mode
if args.mode == "video":
    if not args.path:
        print("Error: Video mode selected but no path provided.")
        exit()
    cap = cv2.VideoCapture(args.path)

elif args.mode == "webcam":
    cap = cv2.VideoCapture(0)  # 0 for the default webcam

elif args.mode == "image":
    if not args.path:
        print("Error: Image mode selected but no path provided.")
        exit()
    img = cv2.imread(args.path)
    if img is None:
        print("Error: Could not open image.")
        exit()

# Function to process frames (used for both video and image input)
def process_frame(frame, display_time=True):
    # Resize the frame to a desired height of 500 pixels while maintaining aspect ratio
    target_height = 1000
    aspect_ratio = frame.shape[1] / frame.shape[0]
    target_size = (int(aspect_ratio * target_height), target_height)
    frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    gray_blurred = cv2.medianBlur(gray, 5)

    # Detect circles (clock face) using HoughCircles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=80,
        minRadius=10,
        maxRadius=400
    )

    if circles is not None:
        # Round the circle parameters and convert to integers
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]

        # Create a mask to isolate the clock face
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        clock_face = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)

        # Convert masked clock face to grayscale for further processing
        clock_face_gray = cv2.cvtColor(clock_face, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detector
        edges = cv2.Canny(clock_face_gray, 100, 200, apertureSize=3)

        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=0.2 * r,
        maxLineGap=12
        )

        detected_time = None  # Initialize detected_time

        if lines is not None:
            # Initialize lists to store angles and lengths
            line_data = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Compute length of the line
                length = math.hypot(x2 - x1, y2 - y1)

                # Compute the angle of the line
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                angle = math.degrees(math.atan2(mid_y - y, mid_x - x))
                if angle < 0:
                    angle += 360

                # Filter lines based on distance from center
                distance_from_center1 = math.hypot(x1 - x, y1 - y)
                distance_from_center2 = math.hypot(x2 - x, y2 - y)
                distance_from_center = min(distance_from_center1, distance_from_center2)
                if distance_from_center > 0.3 * r:
                    continue

                line_data.append({
                    'length': length,
                    'angle': angle,
                    'coords': (x1, y1, x2, y2)
                })

            # Cluster lines by angle similarity
            angle_threshold = 20  # degrees for clustering similar angles
            if len(line_data) > 0:
                clusters = []
                for line in line_data:
                    angle = line['angle']
                    added_to_cluster = False
                    for cluster in clusters:
                        angle_diff = min(abs(angle - cluster['angle']), 360 - abs(angle - cluster['angle']))
                        if angle_diff < angle_threshold:
                            cluster['lines'].append(line)
                            # Recalculate the average angle for the cluster
                            cluster['angle'] = np.mean([l['angle'] for l in cluster['lines']])
                            added_to_cluster = True
                            break
                    if not added_to_cluster:
                        clusters.append({'angle': angle, 'lines': [line]})

                # Calculate representative lines
                representative_lines = []
                for cluster in clusters:
                    avg_length = sum([l['length'] for l in cluster['lines']]) / len(cluster['lines'])
                    avg_angle = sum([l['angle'] for l in cluster['lines']]) / len(cluster['lines'])
                    representative_lines.append({'length': avg_length, 'angle': avg_angle})

                # Determine clock hands based on line lengths
                sorted_lines = sorted(representative_lines, key=lambda l: l['length'])
                hour_line = sorted_lines[0]
                other_lines = sorted_lines[1:]

                # Calculate time
                hour = (hour_line['angle'] / 30 + 3) % 12
                if len(other_lines) >= 2:
                    # If both minute and second hands are detected
                    minute_line,second_line  = other_lines[:2]
                    minute = (minute_line['angle'] % 360) / 6 + 15
                    second = (second_line['angle'] % 360) / 6 + 15
                    minute %= 60
                    second %= 60
                elif len(other_lines) == 1:
                    # If only minute hand is detected
                    minute_line = other_lines[0]
                    minute = (minute_line['angle'] % 360) / 6 + 15
                    minute %= 60
                    second = 0
                else:
                    # If no other hand is detected
                    minute = 0
                    second = 0

                detected_time = f"{int(hour)}:{int(minute)}:{int(second)}"
                print(f"Detected Time: {detected_time}")

                # Draw representative lines on the clock face
                for line in representative_lines:
                    # Compute the coordinates of the representative line
                    angle_rad = math.radians(line['angle'])
                    end_x = int(x + line['length'] * math.cos(angle_rad))
                    end_y = int(y + line['length'] * math.sin(angle_rad))
                    # Draw the line on the clock face
                    cv2.line(clock_face, (x, y), (end_x, end_y), (0, 255, 0), 2)

                # Overlay the detected time on the top-left corner
                if display_time and detected_time:
                    cv2.putText(
                        clock_face,
                        f"Time: {detected_time}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

            else:
                print("No lines detected.")
                if display_time:
                    cv2.putText(
                        clock_face,
                        "Time: N/A",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

        # Display only the clock face with the overlaid time
        cv2.imshow("Clock Face", clock_face)
    else:
        print("No clock face detected.")
        # Optionally, display a black image or the original frame
        black_image = np.zeros_like(frame_resized)
        if display_time:
            cv2.putText(
                black_image,
                "Time: N/A",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        cv2.imshow("Clock Face", black_image)
        # Alternatively, to show the original frame when no clock is detected:
        # cv2.imshow("Clock Face", frame_resized)

if args.mode == "image":
    process_frame(img)
    cv2.waitKey(0)  # Wait for key press to close the window
else:
    if not cap.isOpened():
        print("Error: Could not open video or webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read the frame.")
            break

        process_frame(frame)

        # Exit on 'q' key press
        # time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
cv2.destroyAllWindows()
