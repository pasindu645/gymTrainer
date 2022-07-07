import cv2
import time
import numpy as np
import movementModule as GymMod

cap = cv2.VideoCapture("resources/exercises2.mp4")

Detector = GymMod.poseDetector()
count = 0
dir = 0
pTime = 0
p=0

while True:

    success,img = cap.read()
    img = cv2.resize(img,(1280,670))
    #img = cv2.imread("resources/255-2559636_gym-man-png-men-fitness-png.png")
    img = Detector.findPose(img,False)
    lmlist = Detector.findPosition(img,False)
    #print(lmlist)
    if len(lmlist) != 0:
        #print(lmlist)
        #Detector.Angle(img,12,14,16) #right arm
        angle = Detector.Angle(img,11,13,15) #left arm
        per = np.interp(angle,(200,270),(0,100))
        bar = np.interp(per,(0,100),(650,100))
        #print(per,angle)

        #count up and down terms
        color = (0,255,0)
        if per == 100:
            color = (0, 0, 255)
            if dir ==0:
                count+=0.5
                dir =1

        if per == 0:
            color = (255, 255, 255)
            if dir ==1:
                count+=0.5
                dir=0

        print(count)


        #draw bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color,3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color,cv2.FILLED)
        cv2.rectangle(img, (1100, 15), (1240, 90), (200, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%',(1100,75),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(200,200,0),2)
        # if count==4:
        #     cv2.putText(img, f'Exellent', (550, 300), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,5, (200, 200, 0), 2)




        #draw count
        cv2.rectangle(img,(0,200),(150,400),(0,255,0),cv2.FILLED)
        cv2.putText(img, str(int(count)), (30,350), cv2.FONT_HERSHEY_PLAIN, 10,(200, 0, 0), 6)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("img",img)
    cv2.waitKey(1)


