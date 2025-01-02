import cv2
import numpy as np

# Path to video and image files
video_path = 'task1.mp4'  # Replace with the path to your video
image_path = 'input.png'   # Replace with the path to your image
output_video_path = 'task1_output_sample.avi'
output_image_path = 'output_image.png'

# Display name on video
name_text = "521H0320_LETRI"  # Replace with your name
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)
font_thickness = 2
text_position = (10, 30)

# === Task 1: Video processing ===
cap = cv2.VideoCapture(video_path)  # Load video for processing
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec for output video
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the frames per second from video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video frame width
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video frame height
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))  # Create a video writer for output

while cap.isOpened():
    ret, frame = cap.read()  # Read each frame
    if not ret:
        break  # Exit if no more frames

    # Light balance using CLAHE
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  # Convert to YCrCb color space
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))  # Create CLAHE object for brightness balance
    ycrcb_frame[:, :, 1] = clahe.apply(ycrcb_frame[:, :, 1])  # Apply CLAHE to the Y channel
    balanced_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)  # Convert back to BGR color space

    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2HSV)  # Convert to HSV for color masking
    hsv_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0)  # Apply Gaussian blur to reduce noise

    # Color threshold for blue traffic signs
    lower_blue1 = np.array([100, 100, 100])
    upper_blue1 = np.array([110, 255, 255])
    lower_blue2 = np.array([110, 100, 100])
    upper_blue2 = np.array([130, 255, 255])
    blue_mask1 = cv2.inRange(hsv_frame, lower_blue1, upper_blue1)  # Mask for light blue range
    blue_mask2 = cv2.inRange(hsv_frame, lower_blue2, upper_blue2)  # Mask for dark blue range
    blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)  # Combine both blue masks

    # Color threshold for red traffic signs
    lower_red1 = np.array([0, 120, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 80])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)  # Mask for lower red range
    red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)  # Mask for upper red range
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)  # Combine both red masks
    combined_mask = cv2.bitwise_or(blue_mask, red_mask)  # Combine blue and red masks

    # Remove small noises in the mask
    combined_mask = cv2.erode(combined_mask, None, iterations=1)
    combined_mask = cv2.dilate(combined_mask, None, iterations=2)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

    output_frame = frame.copy()  # Make a copy for drawing

    for contour in contours:
        area = cv2.contourArea(contour)  # Calculate contour area
        if area > 2100:
            circularity = 4 * np.pi * (area / (cv2.arcLength(contour, True) ** 2))  # Calculate circularity
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding box coordinates
            aspect_ratio = w / float(h)
            # Check if it matches a traffic sign shape and size
            if 0.65 < circularity < 1.35 and 0.8 < aspect_ratio < 1.2 and 30 <= w <= 100 and 30 <= h <= 100:
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

    cv2.putText(output_frame, name_text, text_position, font, font_scale, font_color, font_thickness)  # Add name
    out.write(output_frame)  # Write frame to output video

    cv2.imshow("Traffic Sign Detection", output_frame)  # Display frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Stop if 'q' is pressed

cap.release()
out.release()
cv2.destroyAllWindows()

# === Task 2: Image processing ===
img = cv2.imread(image_path)  # Read the input image from the specified path
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

# Get the dimensions of the grayscale image (height and width)
height, width = img_gray.shape

# Split the grayscale image into two halves: top and bottom
top = img_gray[0:height // 2, 0:width]  # The top half of the grayscale image
bottom = img[height // 2:height, 0:width]  # The bottom half in original color

# Define the left and right part ratios for splitting
left_ratio = 4 / 10  # 40% for the left part
right_ratio = 6 / 10  # 60% for the right part

# Calculate the pixel width for each part based on the ratios
left_width = int(width * left_ratio)  # Width of the left part
right_width = int(width * right_ratio)  # Width of the right part

# Split the bottom half into left and right parts
left_part = bottom[:, :left_width]  # The left section of the bottom half
right_part = bottom[:, left_width:]  # The right section of the bottom half

# Image processing function using a simple threshold
def process_image(part_img, kernel_size=(1, 1)):
    # Apply binary inverse thresholding
    _, th1 = cv2.threshold(part_img, 120, 255, cv2.THRESH_BINARY_INV)

    # Create a kernel for morphological operations
    kernel = np.ones(kernel_size, np.uint8)
    
    # Apply morphological opening to remove small noises
    img_open = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    
    # Apply morphological closing to fill gaps
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the processed image
    contours, _ = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert the original part image to color for drawing
    processed_part = cv2.cvtColor(part_img, cv2.COLOR_GRAY2BGR)

    # Loop through each contour to draw bounding boxes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Get the bounding box for each contour
        if 9 < w < 200 and 30 < h < 200:  # Filter contours by width and height
            cv2.rectangle(processed_part, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle

    return processed_part

# Function to detect digits and draw boxes using a simple threshold
def process_image_1(image, kernel_size=(1, 1)):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thres = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)  # Apply binary inverse threshold

    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)  # Apply morphological closing

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through contours to draw bounding boxes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Get the bounding box for each contour
        if 9 < w < 200 and 30 < h < 200:  # Filter based on size
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle

    return image

# Another image processing function using a different threshold value
def process_image_2(image, kernel_size=(1, 1)):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thres = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)  # Apply binary inverse threshold

    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)  # Apply morphological closing

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through contours to draw bounding boxes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Get the bounding box for each contour
        if 9 < w < 200 and 30 < h < 200:  # Filter based on size
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle

    return image

# Process each part of the image with different functions
processed_top = process_image(top)  # Process the top part using the first function
processed_left_part = process_image_1(left_part)  # Process the left part using the second function
processed_right_part = process_image_2(right_part)  # Process the right part using the third function

# Combine the processed images into a final result
final_result = np.vstack((processed_top, np.hstack((processed_left_part, processed_right_part))))

# Display the final result with bounding boxes
cv2.imshow('Rectangles surrounding each digit', final_result)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close all OpenCV windows

# Save the final result to a file
output_path = 'output_image.png'
cv2.imwrite(output_path, final_result)  # Save the image with bounding boxes to the specified file
