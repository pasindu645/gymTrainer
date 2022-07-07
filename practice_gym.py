import mediapipe as mp
import time
import cv2
import csv
import os
import numpy as np
import pandas as pd
import faceMeshMod as fm
import pickle

class_name = "sad"
cap = cv2.VideoCapture(0)
pTime = 0
detector = fm.FaceMesh()
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:



    with open('rfmode_body_languagetest.pkl', 'rb') as f:# rc mode is does not match for this
        model = pickle.load(f)
        print(model)


    while True:
        success, img = cap.read()
        # img = cv2.flip(img, 1)
        img = cv2.resize(img, (720, 560))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, faces = detector.findPosition(img)

        results = holistic.process(imgRGB)

        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                  # mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1)
                                  )


        # # 2. Right hand
        # mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        #                           mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        #                           )

        # # 3. Left Hand
        # mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        #                           mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        #                           )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )



        # num_coords = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)


        landmarks = ['class']
        for val in range(1, 501 + 1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        # with open('test9.csv', mode='w', newline='') as f:
        #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     csv_writer.writerow(landmarks)

        try:
            # Extract Pose landmarks

            pose = results.pose_landmarks.landmark
            # print(pose)
            pose_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            # print(pose_row)

            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # Concate rows
            row = pose_row + face_row




            X = pd.DataFrame([row])
            rfmode_body_language_class = model.predict(X)[0]
            rfmode_body_language_prob = model.predict_proba(X)[0]
            print(rfmode_body_language_class, rfmode_body_language_prob)

            # Append class name
            # row.insert(0, class_name)
            # print(row)



            coords = tuple(np.multiply(
                np.array(
                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                , [720, 520]).astype(int))
            # print(coords)

            # cv2.rectangle(img,
            #               (coords[0], coords[1] + 5),
            #               (coords[0] + len(body_language_class) * 20, coords[1] - 30),
            #               (245, 117, 16), -1)
            if rfmode_body_language_prob[np.argmax(rfmode_body_language_prob)] > 0.70:

                cv2.putText(img, rfmode_body_language_class, coords,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display Class
            cv2.putText(img, 'CLASS'
                        , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, rfmode_body_language_class.split(' ')[0]
                        , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Display Probability
            cv2.putText(img, 'PROB'
                        , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(round(rfmode_body_language_prob[np.argmax(rfmode_body_language_prob)], 2))
                        , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Export to CSV
            # with open('test9.csv', mode='a', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(row)



        except:
            pass










        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (620, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)