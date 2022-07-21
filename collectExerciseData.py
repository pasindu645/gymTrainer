import cv2
import numpy as np
import os
import mediapipe as mp

cap = cv2.VideoCapture('resources/hammer_curls2.mp4')
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data1')

# Actions that we try to detect
# actions = np.array(['dumbell curls', 'push-ups','squatting' ])
# actions = np.array([ 'pushup' , 'squatting', 'hammer_curls'])
actions = np.array([ 'hammer_curls' ])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
# start_folder = 30

for action in actions:
    # dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
    for sequence in range(1, no_sequences + 1):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    for action in actions:
            # Loop through sequences aka videos
            for sequence in range(1, no_sequences + 1):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    success, img = cap.read()
                    img = cv2.resize(img, (1000, 600))
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    results = holistic.process(imgRGB)
                    # print(results)
                    # img, faces = detector.findPosition(img)

                    # 4. Pose Detections
                    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)

                                              )
                    if frame_num == 0:
                        # cv2.putText(img, 'STARTING COLLECTION', (120, 200),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(img, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', img)
                        cv2.waitKey(1)
                    else:
                        cv2.putText(img, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', img)
                        cv2.waitKey(1)

                        # NEW Export keypoints

                    pose = np.array([[indox.x, indox.y, indox.z, indox.visibility] for indox in
                                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(
                        132)

                    keypoints = pose
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)



                    # Break gracefully
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
    cap.release()
    cv2.destroyAllWindows()

