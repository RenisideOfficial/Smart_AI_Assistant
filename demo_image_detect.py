import cv2
from ultralytics import YOLO
from pathlib import Path
import pyttsx3
import queue
import threading
from personal_object import get_personal_objects

class ImageDetect:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.speech_queue = queue.Queue()
        self.voices = self.engine.getProperty("voices")
        self.engine.setProperty('voice', self.voices[0].id)
        
        # Start a single, dedicated thread for speech processing
        threading.Thread(target=self._speech_processor_thread, daemon=True).start()
        
        self.personal_objects = get_personal_objects()

    def _speech_processor_thread(self):
        """Dedicated thread to process speech commands from the queue."""
        while True:
            text = self.speech_queue.get()
            if text is None:  # Sentinel value to stop the thread
                break
            self.engine.say(text)
            self.engine.runAndWait()
            self.speech_queue.task_done()

    def say(self, text):
        """Puts a text command into the speech queue."""
        self.speech_queue.put(text)

    def custom_object(self, label, personal_objects):
        if label in personal_objects.keys():
            text = f"{label} Detected"
        else:
            text = f"An object of type {label} was found"
        return text

    def predict_and_output(self, image_path: str, model_path):
        model = YOLO(model_path)
        frame = cv2.imread(str(image_path))
        results = model.predict(frame, conf=0.5)
        self.process_result(results, frame)
        # Add a sentinel value to the queue to stop the thread when done
        self.speech_queue.put(None) 
        self.speech_queue.join()

    def process_result(self, results, frame):
        print("This is it: ", results, frame)
        if not results:
            # It's better to log this or handle it gracefully
            print("No objects detected.")
            # Don't raise an exception here, as it will crash the program.
            # You can simply return or show the original image.
        else:
            for result in results:
                boxes = result.boxes
                class_names = result.names

                for box in boxes:
                    # get cordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # get confidence
                    confidence = float(box.conf[0])
                    # get id to locate object name
                    class_id = int(box.cls[0])
                    label = class_names[class_id]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)

                    text = self.custom_object(label, self.personal_objects)
                    self.say(text) # Use the correct method to put text in the queue

            self.speech_queue.join() # Wait for all speech to be processed before showing the image

        resized_frame = cv2.resize(frame, (680, 720))

        cv2.imshow("Yolo Detection", resized_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # get path
    image_path = Path(__file__).resolve().parent / "image.jpg"
    model_path = Path(__file__).resolve().parent / "yolov8m.pt"

    # instantiate and detect
    image_detector = ImageDetect()
    image_detector.predict_and_output(image_path, model_path)