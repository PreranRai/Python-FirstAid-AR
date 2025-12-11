# Automated Wound Detection & Augmented Treatment

## 🎥 Demo Video
**[Click here to watch the Project Demo Video](https://drive.google.com/file/d/1sKFPbOd-7wxGxvvPEWg0vmmmSX86pCzb/view?usp=drive_link)**

---

## 📖 Project Overview
This project is a Computer Vision application developed in Python. It utilizes the system's webcam to detect skin wounds in real-time and digitally overlays a first-aid bandage (band-aid) in the correct position and orientation.

## ✨ Features
* **Live Webcam Input:** Captures real-time video feed for dynamic arm positioning.
* **HSV Color Detection:** Robustly identifies wounds based on red hue thresholds (Ranges 0-10 & 170-180).
* **Automated Scaling & Rotation:** Uses `cv2.minAreaRect` to calculate the exact angle and size of the wound to fit the band-aid naturally.
* **Alpha Blending:** Overlays a transparent PNG band-aid onto the video frame without obscuring the background.
* **Debug Mode:** Visualizes the detection area with a bounding box to help users adjust lighting.

## 🛠️ Tech Stack & Prerequisites
* **Language:** Python 3.x
* **Libraries:**
  * `opencv-python` (Computer Vision)
  * `numpy` (Matrix operations)

## ⚙️ Installation

1.  **Clone or Download** this project folder.
2.  **Install Dependencies** using pip:
    ```bash
    pip install opencv-python numpy
    ```
3.  **Verify Assets:** Ensure the following files are in the root directory:
    * `assignment.py` (The main script)
    * `bandaid.png` (Transparent band-aid image)

## 🚀 How to Run

1.  Open your terminal or command prompt.
2.  Navigate to the project directory:
    ```bash
    cd path/to/Wound_Project
    ```
3.  Run the script:
    ```bash
    python assignment.py
    ```

## 📝 Usage Instructions

1.  **Start the Program:** Upon running the script, a webcam window will open.
2.  **Position the Wound:** Place your arm in the frame. Ensure the wound (or red marker) is clearly visible and well-lit.
    * *Tip: Avoid backgrounds with red/brown tones (like wooden walls) to prevent false detection.*
3.  **Capture:** Press **SPACEBAR** to capture the image.
4.  **View Results:** The program will display:
    * **Captured Photo:** The original image.
    * **Debug Window:** Shows the detection area (Green Box).
    * **Result:** The final image with the band-aid applied.
5.  **Exit:** Press **0** (Zero) to close the result windows. Press **Q** to quit the camera feed without capturing.

## 🔧 Troubleshooting

* **Crash with `AttributeError: module 'numpy' has no attribute 'int0'`:**
    * This project uses `np.int32` to ensure compatibility with newer NumPy versions.
* **Band-aid is in the wrong place / Giant Green Box:**
    * Check the "Debug" window. If the green box is capturing the background (e.g., a wooden wall or curtains), try moving to a neutral background (white/grey wall) or holding the arm closer to the camera so the wound is the largest red object.

## 👤 Author
[Your Name]