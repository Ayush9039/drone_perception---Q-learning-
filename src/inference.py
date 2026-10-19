import cv2
import os
import torch
import warnings

# --- Relative Import Fix (src folder ya project root dono jagah se run karne ke liye) ---
try:
    from q_learning import DroneTrafficQAgent
except ImportError:
    from src.q_learning import DroneTrafficQAgent

# --- PYTORCH FIX: Bypass weights_only restriction ---
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load
# ---------------------------------------------------

from ultralytics import YOLO

def run_inference():
    # 1. Project root aur path resolution
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Model path check (root directory aur local dono support karega)
    model_path = os.path.join("models", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "models", "best.pt")

    if not os.path.exists(model_path):
        print(f"Error: {model_path} file nahi mili! Check karein ki models folder me best.pt maujood hai.")
        return
        
    print(f"Model load ho raha hai: {model_path}")
    model = YOLO(model_path)

    # 2. Test video path check
    video_name = "drone_road_video.mp4"
    video_path = os.path.join("test_videos", video_name)
    if not os.path.exists(video_path):
        video_path = os.path.join(base_dir, "test_videos", video_name)

    if not os.path.exists(video_path):
        print(f"Error: {video_path} file nahi mili! Check karein ki test_videos folder me video sahi naam se hai.")
        return

    cap = cv2.VideoCapture(video_path)
    print("Inference & Q-Learning shuru ho raha hai... Window close karne ke liye 'q' dabayein.")

    # 3. Tracking aur Q-Learning agent initialize karna
    q_agent = DroneTrafficQAgent(lr=0.1, gamma=0.9, epsilon=0.2)
    current_state = 0
    id_map = {} 
    class_counters = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # YOLOv8 tracking (persist=True se frames me continuous track id milti hai)
        results = model.track(frame, persist=True, conf=0.35, verbose=False)
        
        boxes = results[0].boxes
        vehicle_count = 0

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            xyxy = boxes.xyxy.int().cpu().tolist()
            names = model.names
            vehicle_count = len(track_ids)

            for box, track_id, cls_id in zip(xyxy, track_ids, classes):
                class_name = names[cls_id]

                # Har vehicle ko sequential label dena (jaise: car 1, car 2, van 1)
                if track_id not in id_map:
                    class_counters[class_name] = class_counters.get(class_name, 0) + 1
                    id_map[track_id] = f"{class_name} {class_counters[class_name]}"

                custom_label = id_map[track_id]

                # Bounding box draw karna
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)
                
                # Label box aur text draw karna (confidence numbers ke bina)
                (w, h), _ = cv2.getTextSize(custom_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + w, y1), (0, 255, 200), -1)
                cv2.putText(frame, custom_label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 4. Q-Learning Execution Step
        next_state = q_agent.get_state(vehicle_count)
        action = q_agent.choose_action(current_state)
        reward = q_agent.get_reward(current_state, action)
        q_agent.update(current_state, action, reward, next_state)
        current_state = next_state

        # State aur Action ke human-readable names
        state_names = {0: "Low", 1: "Medium", 2: "High"}
        action_names = {0: "Maintain", 1: "Expand", 2: "Alert"}

        # Screen overlay (Live Dashboard)
        overlay_text = f"Vehicles: {vehicle_count} | State: {state_names[current_state]} | Action: {action_names[action]} | Reward: {reward}"
        cv2.rectangle(frame, (10, 10), (620, 50), (30, 30, 30), -1)
        cv2.putText(frame, overlay_text, (20, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Output frame screen par dikhana
        cv2.imshow("Drone Perception System - Q-Learning", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 5. Q-Table aur learning logs ko JSON file me save karna
    output_json = os.path.join(base_dir, "q_learning_results.json")
    q_agent.save_to_json(output_json)
    print("Inference aur Q-Learning process complete!")

if __name__ == "__main__":
    run_inference()