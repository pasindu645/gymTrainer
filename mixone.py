import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.7)


cap = cv2.VideoCapture("resources/Basic Dance Steps for Everyone - 3 Simple Moves - Practice Everyday - Deepak Tulsyan - Part 8.mkv")
pTime = 0
while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results1 = pose.process(imgRGB)
    results2 = faceDetection.process(imgRGB)
    #print(results.pose_landmarks)

    if results1.pose_landmarks:
        mpDraw.draw_landmarks(img, results1.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results1.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    if results2.detections:
        for id, detection in enumerate(results2.detections):
            #mpDraw.draw_detection(img, detection)
            #print(id,detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
             2, (255, 0, 255), 2)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)