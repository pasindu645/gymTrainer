import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

cap = cv2.VideoCapture('resources/exercises1.mp4')
pTime = 0


with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:

    DATA_PATH = os.path.join('MP_Data')

    # Actions that we try to detect
    # actions = np.array(['dumble curls'])
    actions = np.array(['push-ups', 'dumble curls', 'hammer curls'])

    # Thirty videos worth of data
    no_sequences = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30

    # Folder start
    start_folder = 30
    # for action in actions:
    #     # dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
    #     for sequence in range(1, no_sequences + 1):
    #         try:
    #             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
    #         except:
    #             pass


    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(1, no_sequences + 1):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):


                # while True:
                    success, img = cap.read()
                    img = cv2.resize(img, (1000, 720))
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    results = holistic.process(imgRGB)
                    # print(results)
                    # img, faces = detector.findPosition(img)

                    # 4. Pose Detections
                    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)

                                              )

                    # mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                    #                           mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                    #                           mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                    #                           )

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

                    if frame_num == 0:
                        cv2.putText(img, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(img, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', img)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(img, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', img)

                    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(
                        132)
                    # print(pose)



                    # npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    # np.save(npy_path, pose)


                    cv2.waitKey(1)