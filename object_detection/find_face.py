# by Hatice Candan
# face and eye detection from video camera

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Haar trained model - only frontal faces detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


while(True):
    ret, frame = cap.read() # read frame by frame
    # A frame is an array of 3 matrices (RGB)

    # convert the image into gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # img : gray scale image 
    # Scaled factor: scaled factor determines the ratio at which the image is resized during the face detection process
    # minNeighbors: specifies the minimum number of neighboring rectangles required for a detected region to be considered a face

    # face_cascade.detectMultiScale(img, scaled factor, minNeighbors)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    kernel_size = (15,15)

    # draw a purple (160,32,240) rectangle if the faces are detected
    for (xe, ye, we, he) in eyes:
        cv2.rectangle(frame, (xe, ye), (xe+we, ye+he), (0, 255, 0), 2)
        rect_eye = (2*(we+he))
        # write faces kernel size on image
        text = "Eye " + str(rect_eye)
        cv2.putText(frame, text, (xe, ye-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    for (x,y,w,h) in faces:
      face_roi = frame[y:y+h, x:x+w]

      #if you want to blurring face using it
      #blurred_roi = cv2.blur(face_roi, kernel_size)
      #frame[y:y+h, x:x+w] = blurred_roi
        
      cv2.rectangle(frame, (x, y), (x+w, y+h), (160 , 32, 240), 2)
      rect = (2*(w+h))
      # write faces kernel size on image
      text = "Face kernel size = " + str(rect)
      cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160 , 32, 240), 2)
    cv2.imshow('Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):#q terminates script
        break

# destroy all opened windows
cap.release()
cv2.destroyAllWindows()