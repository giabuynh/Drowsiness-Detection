from imutils import face_utils
import dlib
import imutils
import cv2
from calculator_functions import *
import re

def testing_video():
  detector = dlib.simple_object_detector("model/detector.svm")
  detector = dlib.get_frontal_face_detector()
  cap = cv2.VideoCapture(0)

  while True:
      ret, frame = cap.read()

      cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

      frame = imutils.resize(frame, width=500)
      (h, w) = frame.shape[:2]
      rects = detector(frame, 0)
      if len(rects) > 0:
        rect = get_max_area_rect(rects)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      cv2.imshow("output", frame)

      # detect any kepresses
      key = cv2.waitKey(1) & 0xFF
      
      # set the last active check time as current time
      # if the `q` key was pressed, break from the loop
      if key == ord("q"):
          break

def testing_img():
  dataset_path = "../ibug_300W_large_face_landmark_dataset/"
  file = open(dataset_path + "face_mask_training_from_root_gb.xml")
  detector = dlib.simple_object_detector("model/detector.svm")
  detector = dlib.get_frontal_face_detector()
  for line in file.readlines():
    if len(re.findall("file='", line)) > 0:
      image_file = line[line.find("file='") + 6:line.find("'>")]
      frame = cv2.imread(image_file)

      cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

      frame = imutils.resize(frame, width=500)
      (h, w) = frame.shape[:2]
      rects = detector(frame, 0)
      if len(rects) > 0:
        rect = get_max_area_rect(rects)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      cv2.imshow("output", frame)

      cv2.waitKey(0)

testing_video()
# testing_img()