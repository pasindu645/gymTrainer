import cv2
import mediapipe as mp
import time
# import movementModule as move

class FaceMesh():
    def __init__(self, static_image_mode=False, max_num_faces=2,min_detection_confidence=0.5,
                        min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,self.max_num_faces,
                                                 self.min_detection_confidence,self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)


    def findPosition(self, img, draw=True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            #for id, lm in enumerate(self.faceLms.landmark):
                for faceLms in self.results.multi_face_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec,
                                                   self.drawSpec)

                    for id, lm in enumerate(faceLms.landmark):
                            #print(lm)
                            ih, iw, ic = img.shape
                            x, y = int(lm.x * iw), int(lm.y * ih)
                            #print(id, x, y)
                            faces.append(id)
                            faces.append(x)
                            faces.append(y)
                            faces.append(lm.z)


        return img,faces





def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMesh(max_num_faces=2)
    # detector2 = move.poseDetector()





    while True:
        success, img = cap.read()
        img,faces = detector.findPosition(img)
        # img = detector2.findPose(img)



        if len(faces) !=0:
            print(faces)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)





if __name__ == "__main__":
    main()