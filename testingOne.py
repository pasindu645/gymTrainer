import cv2
import numpy as np
import os
import csv
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import testModule as tm

sequence1 = []
sentence = []
predictions = []
threshold = 0.5
# trainedData = tm

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
# actions = np.array(['dumble curls'])
actions = np.array(['dumble curls', 'push-ups','squatting' ])

# Thirty videos worth of data
no_sequences = 30
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}
print(label_map)
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        a = np.array(sequences).shape
        # print(a)

X = np.array(sequences)
print(X.shape)
y = to_categorical(labels).astype(int)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
b = X_train.shape
print(b)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=80, callbacks=[tb_callback])

c = model.summary()
print(c)

res = model.predict(X_test)
d = actions[np.argmax(res[4])]

e = actions[np.argmax(y_test[4])]
print(d)
print(e)

model.save('exercisesaction10.h5')
model.load_weights('exercisesaction10.h5')

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

f = multilabel_confusion_matrix(ytrue, yhat)

g = accuracy_score(ytrue, yhat)

print(f)
print(g)

cap = cv2.VideoCapture('resources/pushup1.mp4')

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:

        success, img = cap.read()
        # img = cv2.resize(img, (1000, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = holistic.process(imgRGB)
        # print(results)


        # 4. Pose Detections
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)

                                  )
        pose = np.array([[indox.x, indox.y, indox.z, indox.visibility] for indox in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(
            132)





        # 2. Prediction logic
        keypoints = pose
        sequence1.append(keypoints)
        sequence1 = sequence1[-30:]

        if len(sequence1) == 30:

            res = model.predict(np.expand_dims(sequence1, axis=0))[0]
            print(actions[np.argmax(res)])
            print(res[np.argmax(res)])
            predictions.append(np.argmax(res))
            # print(predictions)

            # 3. Viz logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            # image = prob_viz(res, actions, image, colors)

        cv2.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
        # if (res[np.argmax(res)]) > 0.97:
        # cv2.putText(img, (actions[np.argmax(res)]), (3, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', img)

        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()