"""
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
"""
import cv2
import numpy as np
import os
import datetime

con_data = 0

called = False

cmd = os.popen("scp pi@1other pi ip:/home/pi/trainer/*.yml /home/pi/trainer")
cmd.read()
print("\n [INFO] Trainer File Load Successfully")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/home/pi/trainer/trainer.yml")
cascadePath = "/home/pi/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# names related to ids: example ==> Name 1: id=1,  etc
# names = ['None', '1', '2', '3', '4', '5']
names = [
    "None",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
]


# Initialize and start realtime video capture
cam = cv2.VideoCapture(
    "rtsp://IP Adress:Port/user=admin&password=admin@123&channel=1&stream=1.sdp?"
)
# cam = cv2.VideoCapture('rtsp://IP Adress:Port/user=admin&password=admin@123$&channel=3&stream=1.sdp?')
# cam = cv2.VideoCapture(0)

cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
cam.set(cv2.CAP_PROP_FPS, 10)

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
# Percentage
in_min = 0
in_max = 100
out_min = 100
out_max = 0


while True:
    ret, img = cam.read()
    # img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y : y + h, x : x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if confidence < 100:
            id = names[id]
            con_data = round(confidence)
            print("[INFO] User ID: " + str(id) + " Confidence: " + str(con_data) + "%")
            # confidence = "  {0}%".format(round(100 - confidence))
            if not called:
                cmd = os.popen('aplay "/home/pi/thank you.wav"')  # Play Sound File
                # cmd.read()
                called = True
                break

        else:
            id = 0
            con_data = round(confidence)
            # confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        # cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
        cv2.putText(img, str(con_data), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    cv2.imshow("camera", img)

    k = cv2.waitKey(1) & 0xFF  # Press 'ESC' for exiting video
    if k == 27:
        break


# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
