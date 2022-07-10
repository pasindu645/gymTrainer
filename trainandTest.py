import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

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
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

    c = model.summary()
    print(c)

    res = model.predict(X_test)
    d = actions[np.argmax(res[4])]

    e = actions[np.argmax(y_test[4])]
    print(d)
    print(e)

    model.save('action.h5')

    yhat = model.predict(X_train)
    ytrue = np.argmax(y_train, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    multilabel_confusion_matrix(ytrue, yhat)

    accuracy_score(ytrue, yhat)

    # while True:
    #     success, img = cap.read()
    #     img = cv2.resize(img, (1000, 720))
    #     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
