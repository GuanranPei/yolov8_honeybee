from ultralytics import YOLO
import os

# Step 1: Load the trained model from the specified folder
model = YOLO('/home/guanran/robotic_flower/bee_detection/yolov8/runs/detect/yolov8n-honeybee/weights/best.pt')

# Step 2: Get all image files from the directory
image_directory = '/home/guanran/robotic_flower/bee_detection/yolov8/test/images'
image_files = [os.path.join(image_directory, img) for img in os.listdir(image_directory) if img.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Step 3: Make predictions on all images
results = model.predict(source=image_files[0])

results[0].save_txt("/home/guanran/robotic_flower/bee_detection/yolov8/results.csv")

# # Step 4: Visualize each result
# for result in results:
#     result.show()