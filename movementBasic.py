import cv2
import mediapipe as mp
import time
import faceMeshMod as fm
import csv
import os
import numpy as np

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# "resources/Basic Dance Steps for Everyone - 3 Simple Moves - Practice Everyday - Deepak Tulsyan - Part 8.mkv"
cap = cv2.VideoCapture(0)
pTime = 0
detector = fm.FaceMesh()
while True:
    success , img = cap.read()
    img = cv2.resize(img, (1180, 670))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = detector.findPosition(img)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape

            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist =[]
            lmlist.append([id, cx, cy])
            if len(lmlist) != 0:
                print(lmlist)
                with open('pasi.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(lmlist)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)