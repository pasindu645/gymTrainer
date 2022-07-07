import cv2
import mediapipe as mp
import time
import numpy as np
import csv
# import faceMeshMod as fm
import pandas as pd

import pickle


cap = cv2.VideoCapture('resources/exercisesall.mp4')
pTime = 0
class_name = "sad"
# detector = fm.FaceMesh()

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:

    with open('lrmode_body_exercise.pkl', 'rb') as f:
        model = pickle.load(f)
        print(model)
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1000, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = holistic.process(imgRGB)
        # img, faces = detector.findPosition(img)

        # 4. Pose Detections
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)

                                  )
        lm_face = []
        for index in range(1, 469):
            lm_face += ['lmface{}'.format(index - 1), 'X{}'.format(index), 'Y{}'.format(index),'Z{}'.format(index)]

        lm_pose = ['class']
        for val in range(1, 34):
            lm_pose += ['lmpose{}'.format(val - 1), 'X{}'.format(val), 'Y{}'.format(val),'Z{}'.format(val)]
        #
        # with open('test14.csv', mode='w', newline='') as f:
        #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     csv_writer.writerow(lm_pose+lm_face)
        #



        # pose = results.pose_landmarks.landmark
        # print(pose)
        try:

            lmposelist =[]
            if results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    # print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmposelist.append(id)
                    lmposelist.append(cx)
                    lmposelist.append(cy)
                    lmposelist.append(lm.z)

            database = lmposelist
            # print(database)

            X = pd.DataFrame([database])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)



            database.insert(0, class_name)




            # another way to get coordinates of pose landmarks
            # pose_row = list(
            #     np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            # print(pose_row)
            # database.insert(0, class_name)

            # Export to CSV
            # with open('test14.csv', mode='a', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(database)

            # coords = tuple(np.multiply(
            #     np.array(
            #         (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
            #          results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
            #     , [720, 520]).astype(int))
            # print(coords)

            # cv2.rectangle(img,
            #               (coords[0], coords[1] + 5),
            #               (coords[0] + len(body_language_class) * 20, coords[1] - 30),
            #               (245, 117, 16), -1)
            if body_language_prob[np.argmax(body_language_prob)] > 0.57:
                cv2.rectangle(img,(600,120),(870,170),(255,0,255),cv2.FILLED)
                cv2.putText(img, body_language_class, (600,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display Class
            cv2.rectangle(img, (0, 0), (250, 50), (0, 255, 255), cv2.FILLED)
            cv2.putText(img, 'CLASS'
                        , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, body_language_class.split(' ')[0]
                        , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Display Probability
            cv2.putText(img, 'PROB'
                        , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                        , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)









        except:
         pass

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (900, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)