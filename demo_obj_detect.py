import cv2
from pathlib import Path
from ultralytics import YOLO
import pyttsx3

# locate and load the model
model_path = Path(__file__).resolve().parent / "yolov8m.pt"
model = YOLO(model_path)

# initialize speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# open the default webcam
cap = cv2.VideoCapture(0)

# keep detecting in a loop
while True:
    ret, frame = cap.read() # read a frame from the camera
    if not ret: break   # stop if camera fails
    
    # run detection
    results = model.predict(frame, conf=0.5)
    
    for result in results:
        boxes = result.boxes
        class_names = result.names
        
        for box in boxes:
            # extract the cordinate and convert to integer
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # extract the confidence and convert to float
            confidence = float(box.conf[0])
            # extract the class id and use it to get the detected object
            class_id = int(box.cls[0])
            label = class_names[class_id]
            
            # draw the box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # speak the first detected object per frame
            engine.say(f"{label} detected")
            engine.runAndWait()
            
    cv2.imshow("YOLO REAL-TIME DETECTION", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# cleanup
cap.release()
cv2.destroyAllWindows()