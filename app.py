import cv2
from Inference import Inference
import time

def new_show(frame,start_time,end_time):
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the captured frame
    cv2.imshow('Camera', frame)


static_quantized = 'models/static_quantized.onnx'
score=0.5

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
x = Inference(static_quantized)

while True:
    start_time=time.time()
    ret, frame = cam.read()
    y = x.pipeline(frame,score)
    end_time = time.time()
    new_show(frame,start_time,end_time)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()