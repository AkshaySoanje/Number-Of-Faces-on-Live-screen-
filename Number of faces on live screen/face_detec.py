import cv2
import numpy as np
import urllib.request

model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt"
weights_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

urllib.request.urlretrieve(model_url, "deploy.prototxt")
urllib.request.urlretrieve(weights_url, "mobilenet_iter_73000.caffemodel")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
cap = cv2.VideoCapture(0) 
while True:
    ret, frame = cap.read()
    if not ret:
        break
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    k=0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  
            class_id = int(detections[0, 0, i, 1])
            #label = f"Confidence: {confidence:.2f}"
            #This class id 15 is for Face detect
            if class_id == 15:
                k+=1  
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255,0), 2)
            #y = startY - 15 if startY - 15 > 15 else startY + 15
           #cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Video Feed", frame)
    print("Number of People Detected in frame",k)
    if(k>1):
        print("Candidate is having Multiple Faces on screen")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()