import cv2
import urllib
import os
import pdb
import numpy as np

snap = os.environ["SNAP"]
face_cascade = cv2.CascadeClassifier(snap + '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(snap + '/usr/share/opencv4/haarcascades/haarcascade_eye.xml')

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        #self.video = urllib.urlopen('http://192.168.10.12:8080/video')   #cv2.VideoCapture(0)
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def detect_eyes(self, gray, frame):
        """ Input = greyscale image or frame from video stream
            Output = Image with rectangle boxes around eyes and face
        """
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
          cv2.rectangle(frame, (x,y), (x+w, y+h), (32, 84, 233), 2)

          roi_gray = gray[y:y+h, x:x+w]
          roi_color = frame[y:y+h, x:x+w]

          eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)

          for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (111, 33, 119), 2)
            #cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        return frame

    def detect_faces(self, gray, frame):
        """ Input = greyscale image or frame from video stream
            Output = Image with rectangle box in the face
        """
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
          cv2.rectangle(frame, (x,y), (x+w, y+h), (32, 84, 233), 2)

        return frame

    def get_frame(self):
        success, image = self.video.read()
        image_flip = cv2.flip(image,1)

        gray = cv2.cvtColor(image_flip, cv2.COLOR_BGR2GRAY)
        canvas = self.detect_eyes(gray, image_flip)

        ret, jpeg = cv2.imencode('.jpg', image_flip)
        return jpeg.tobytes()
