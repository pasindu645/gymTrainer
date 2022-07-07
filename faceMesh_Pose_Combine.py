import cv2
import mediapipe as mp
import time
import faceMeshMod as fm
import movementModule as move

cap = cv2.VideoCapture("resources/Basic Dance Steps for Everyone - 3 Simple Moves - Practice Everyday - Deepak Tulsyan - Part 8.mkv")
pTime = 0
detector = fm.FaceMesh()
detector2 =move.poseDetector()
mpDraw = mp.solutions.drawing_utils
mpHolistic = mp.solutions.holistic

while True:
    success , img = cap.read()
    img = cv2.resize(img, (1180, 670))
    img, faces = detector.findPosition(img)
    img = detector2.findPose(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)