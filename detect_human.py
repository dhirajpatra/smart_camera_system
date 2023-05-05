# import the necessary packages
import numpy as np
import cv2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

# cv2.startWindowThread()

# open webcam video stream
# Initialize and start realtime video capture from IP camera
# cam = cv2.VideoCapture(
#     "rtsp://IP Adress:Port/user=admin&password=admin@123&channel=1&stream=1.sdp?"
# )
# cam = cv2.VideoCapture('rtsp://IP Adress:Port/user=admin&password=admin@123$&channel=3&stream=1.sdp?')
cap = cv2.VideoCapture(0)

# # the output will be written to output.avi
# out = cv2.VideoWriter(
#     'output.avi',
#     cv2.VideoWriter_fourcc(*'MJPG'),
#     15.,
#     (640,480))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(
        gray, winStride=(4, 4), padding=(4, 4), scale=1.05
    )

    # Drawing the regions in the
    # Image
    for x1, y1, w1, h1 in boxes:
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

    # face detection box
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x2, y2, w2, h2 in faces_rect:
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), thickness=2)

    upper_body_rect = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x3, y3, w3, h3 in faces_rect:
        cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (0, 255, 255), thickness=2)

    # Write the output video
    # out.write(frame.astype("uint8"))
    # Display the resulting frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
# and release the output
# out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
