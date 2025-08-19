import os
import cv2
import torch
# Load the custom-trained YOLO model
model=torch.hub.load('ultralytics/yolov5','custom', path='C:/Users/Rajanna/AppData/Local/Programs/Python/Python312/yolov5/runs/train/exp9/weights/best.pt')
# Define the path to the folder containing your images image_folder = 'C:/Users/Rajanna/AppData/Local/Programs/Python/Python312/dataset/dataset/images' # Replace with your actual folder path
# Get all image paths in the folder
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.jpeg', '.png'))]
if not image_paths:
print("No images found in the specified folder.")
else:
print(f"Found {len(image_paths)} images to process.")
class_names = model.names # Custom class names inferred from your model
road_info = []
for i, image_path in enumerate(image_paths):
image = cv2.imread(image_path)
if image is None:
print(f"Error loading image from {image_path}")
continue
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = model(image_rgb)
detected_objects = results.xyxy[0].numpy()
vehicle_count = len(detected_objects)
congestion = (vehicle_count / 100) * 100 # Adjusted for max capacity
congestion_level = "High" if congestion > 30 else "Moderate" if congestion > 10 else "Low"
# Check for emergency vehicles
emergency_vehicle_detected = any(
class_names[int(obj[5])] == 'emergency' for obj in detected_objects
)
if emergency_vehicle_detected:
time_for_emergency = 100 / 20 # Update with your specific values
signal_time = max(30, time_for_emergency)
priority = 1
signal_status = f"Green for Emergency Vehicle - Signal Time: {signal_time:.2f} seconds"
congestion_level = "Low"
else:
signal_time = {"Low": 15, "Moderate": 30, "High": 45}[congestion_level]
priority = 4 if congestion_level == "High" else 3 if congestion_level == "Moderate" else 2
signal_status = f"{congestion_level} Congestion - Signal Time: {signal_time} seconds"
road_info.append({
"road": i + 1,
"vehicle_count": vehicle_count,
"congestion_level": congestion_level,
"emergency_detected": emergency_vehicle_detected,
"signal_time": signal_time,
"priority": priority,
"status": signal_status
})
cv2.putText(image, f"Signal: {signal_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow(f'Road {i + 1}', image)
road_info.sort(key=lambda x: x["priority"])
for road in road_info:
print(f"Road {road['road']}: {road['status']}")
cv2.waitKey(0)
cv2.destroyAllWindow
