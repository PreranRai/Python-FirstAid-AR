import cv2
import numpy as np

def overlay_transparent(background, overlay, x, y):
    """Overlays a PNG (with alpha channel) onto a background image at (x, y)."""
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    if x >= bg_w or y >= bg_h or x + ov_w < 0 or y + ov_h < 0:
        return background

    bg_x = max(x, 0)
    bg_y = max(y, 0)
    ov_x = max(0, -x)
    ov_y = max(0, -y)
    
    w = min(bg_w - bg_x, ov_w - ov_x)
    h = min(bg_h - bg_y, ov_h - ov_y)

    if w <= 0 or h <= 0:
        return background

    overlay_crop = overlay[ov_y:ov_y+h, ov_x:ov_x+w]
    background_crop = background[bg_y:bg_y+h, bg_x:bg_x+w]

    alpha = overlay_crop[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(3):
        background_crop[:, :, c] = (alpha * overlay_crop[:, :, c] +
                                    alpha_inv * background_crop[:, :, c])

    background[bg_y:bg_y+h, bg_x:bg_x+w] = background_crop
    return background

def rotate_image(image, angle):
    """Rotates an image around its center."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# --- MAIN EXECUTION ---

# 1. Setup Camera and Load Band-Aid
cap = cv2.VideoCapture(0) # 0 is usually the default webcam
bandaid_path = 'bandaid.png' 
bandaid = cv2.imread(bandaid_path, cv2.IMREAD_UNCHANGED)

if bandaid is None:
    print("Error: Could not load 'bandaid.png'. Check the file path.")
    exit()

print("--- INSTRUCTIONS ---")
print("1. Place your arm with the wound in front of the camera.")
print("2. Ensure the wound is well-lit.")
print("3. Press 'SPACE' to take the photo.")
print("4. Press 'Q' to quit without taking a photo.")

captured_frame = None

# 2. Camera Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Show live feed
    cv2.imshow("Camera - Press SPACE to Capture", frame)

    key = cv2.waitKey(1)
    if key % 256 == 32: # SPACE key
        captured_frame = frame
        print("Photo taken!")
        break
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("Camera - Press SPACE to Capture")

# 3. Process the Captured Image
if captured_frame is not None:
    original = captured_frame.copy()
    debug_img = original.copy() 
    
    # Pre-processing & Wound Detection (HSV Color)
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # Red color range 
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = original.copy()
    wound_detected = False

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if valid_contours:
            wound_detected = True
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # --- FIX IS HERE ---
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # <--- CHANGED TO np.int32
            cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
            # -------------------

            (center_x, center_y), (w, h), angle = rect

            # Scale Band-aid
            scale_factor = 2.0  
            wound_size = max(w, h)
            
            b_h, b_w = bandaid.shape[:2]
            ratio = b_w / b_h
            new_w = int(wound_size * scale_factor)
            new_h = int(new_w / ratio)
            
            if new_w > 0 and new_h > 0:
                resized_bandaid = cv2.resize(bandaid, (new_w, new_h))

                if w < h: 
                    angle = angle - 90
                
                rotated_bandaid = rotate_image(resized_bandaid, angle)

                ov_h, ov_w = rotated_bandaid.shape[:2]
                y_pos = int(center_y - ov_h / 2)
                x_pos = int(center_x - ov_w / 2)

                output_img = overlay_transparent(output_img, rotated_bandaid, x_pos, y_pos)

    if not wound_detected:
        print("No wound detected! Try better lighting or a red marker.")
        cv2.putText(output_img, "No Wound Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("DEBUG: Green Box = Detected Area", debug_img)
    cv2.imshow("Result with Band-Aid", output_img)
    
    print("Press '0' (zero) to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()