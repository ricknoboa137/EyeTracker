import cv2
import numpy as np
import pyautogui
import datetime
import os

# ---

## Record Individual Screen Frames

# Specify screen resolution
SCREEN_SIZE = tuple(pyautogui.size())

# Generate a timestamp for the recording session
timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]#strftime("%Y-%m-%d_%H-%M-%S")
output_folder = f"screen_frames_{timestamp}"

# Create the output folder if it doesn't exist
try:
    os.makedirs(output_folder)
    print(f"Created output directory: {output_folder}")
except OSError as e:
    print(f"Error creating directory {output_folder}: {e}")
    exit()

frame_count = 0

print(f"Recording individual frames. Press 'q' to stop recording. Frames saving to: {output_folder}/")

while True:
    # Take screenshot using PyAutoGUI
    img = pyautogui.screenshot()

    # Convert the screenshot to a numpy array
    frame = np.array(img)

    # Convert RGB to BGR (OpenCV uses BGR by default, but pyautogui gives RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_date=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    # Generate a unique filename for each frame
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}_{frame_date}.png") # :06d pads with leading zeros

    # Save the frame as a PNG image
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

    # Optional: Display the recording
    # cv2.imshow("Screen Recorder (Press 'q' to stop)", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all windows (if imshow was used)
cv2.destroyAllWindows()
print(f"Recording stopped. Total frames saved: {frame_count}")
print(f"All frames are saved in the directory: {output_folder}")

# ---

# Optional: You can later combine these frames into a video using another script or a video editing tool.
# Example command line for ffmpeg (assuming frames are named frame_000000.png, frame_000001.png, etc.):
# ffmpeg -framerate 20 -i screen_frames_YYYY-MM-DD_HH-MM-SS/frame_%06d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4