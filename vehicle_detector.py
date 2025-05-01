from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt") 

ALLOWED_CLASSES = [2, 3, 5, 7] 

def detect_vehicles(frame, confidence_threshold=0.3):
    
    results = model.predict(frame, conf=confidence_threshold, verbose=False)
    
    vehicle_counts = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0}
    
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls in ALLOWED_CLASSES:
            name = model.names[cls]
            if name in vehicle_counts:
                vehicle_counts[name] += 1

    # print(vehicle_counts["car"])
    # print(vehicle_counts["bus"])
    # print(vehicle_counts["truck"])
    # print(vehicle_counts["motorbike"])
    
    return vehicle_counts

def score(vehicle_obj):
    return vehicle_obj["car"]*2.5+vehicle_obj["bus"]*3.5+vehicle_obj["truck"]*4.0+vehicle_obj["motorbike"]*1.5
def clamp_score(value):
 
    if value < 10:
        return 10
    elif value > 40:
        return 40
    else:
        return value


summary = {}

for i in range(1, 5):
    frame = cv2.imread(f"frame{i}.png")
    vehicle_obj = detect_vehicles(frame)
    total_score = score(vehicle_obj)
    summary[i] = [vehicle_obj, total_score]


def display_signal_status(current_signal, phase):

    for i in range(1, 5):
        if i == current_signal:
            print(f"Signal {i}: {phase}")
        else:
            print(f"Signal {i}: RED")
    print("-" * 30)


sorted_signals = sorted(summary.items(), key=lambda x: x[1][1], reverse=True)

for signal, (vehicle_obj, traffic_score) in sorted_signals:
    print(f"\nPreparing to clear Signal {signal}...")
    print(f"Vehicles detected: {vehicle_obj}")
    print(f"Traffic Score: {traffic_score}")
    

    for remaining in range(5, 0, -1):
        display_signal_status(signal, phase="YELLOW")
        print(f"Yellow light time left: {remaining} sec\n")
        time.sleep(1)


    green_light_duration = clamp_score(traffic_score)
    for remaining in range(int(green_light_duration), 0, -1):
        display_signal_status(signal, phase="GREEN")
        print(f"Green light time left: {remaining} sec\n")
        time.sleep(1)
    
    print(f"Signal {signal} cleared.\n")
    time.sleep(1)
