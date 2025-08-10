import cv2
import numpy as np
import pyttsx3
import queue
import threading
from ultralytics import YOLO
from personal_object import get_personal_objects

class ObjectDetection:
    def __init__(self, engine, assistant=None):
        self.engine = engine
        self.assistant = assistant
        # Load YOLOv8 model
        self.model = YOLO("yolov8m.pt")
        
        # Queue for managing speech synthesis
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self.process_speech_queue, daemon=True)
        self.speech_thread.start()

        # Personal object database
        self.personal_objects = get_personal_objects()

    def process_speech_queue(self):
        """Threaded speech processor to handle speech synthesis without blocking."""
        while True:
            text = self.speech_queue.get()
            self.engine.say(text)
            self.engine.runAndWait()
            self.speech_queue.task_done()

    def speak(self, text):
        """Add text to the speech queue."""
        self.speech_queue.put(text)

    def detect_objects(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera!")
        else:
            print("Camera is working!")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv8 detection (much simpler API)
            results = self.model(frame, verbose=False)  # Set verbose=False to disable console output
            
            detected_objects = []  # For scene understanding

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]
                    
                    # Only process detections with confidence > 50%
                    if confidence > 0.5:
                        w = x2 - x1
                        h = y2 - y1
                        color_name = self.get_color_name(frame, x1, y1, w, h)
                        position = self.get_relative_position(x1 + w // 2, frame.shape[1])
                        distance = self.calculate_distance(w)

                        description = f"{label} of color {color_name} is {position}, approximately {distance:.2f} cm away."
                        detected_objects.append(description)
                        self.speak(description)

                        # Personal object recognition
                        if label.lower() in self.personal_objects:
                            personal_description = f"{self.personal_objects[label.lower()]} is detected."
                            self.speak(personal_description)

                        # Draw bounding boxes
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, description, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Scene understanding
            if detected_objects:
                scene_description = "The scene contains: " + ", ".join(detected_objects)
                self.speak(scene_description)

            # Display frame
            cv2.imshow('Enhanced Object Detection with YOLOv8', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def calculate_distance(self, width_in_pixels):
        """Calculate distance based on bounding box width."""
        focal_length = 600  # focal length (in pixels)
        real_width = 15  # real width of the object (in cm)
        distance = (real_width * focal_length) / width_in_pixels
        return distance

    def get_relative_position(self, center_x, frame_width):
        """Determine the relative position of an object in the frame."""
        if center_x < frame_width * 0.33:
            return "to your left"
        elif center_x > frame_width * 0.66:
            return "to your right"
        else:
            return "in front of you"

    def get_color_name(self, frame, x, y, w, h):
        """Get the dominant color name of an object."""
        cropped = frame[y:y + h, x:x + w]
        average_color = np.mean(cropped, axis=(0, 1))  # Average BGR color
        color_name = self.map_color(average_color)
        return color_name

    def map_color(self, bgr):
        """Map BGR color to a color name."""
        blue, green, red = bgr
        if red > 150 and green < 100 and blue < 100:
            return "red"
        elif green > 150 and red < 100 and blue < 100:
            return "green"
        elif blue > 150 and red < 100 and green < 100:
            return "blue"
        else:
            return "unknown color"

if __name__ == "__main__":
    engine = pyttsx3.init()
    object_detection = ObjectDetection(engine)
    object_detection.detect_objects()