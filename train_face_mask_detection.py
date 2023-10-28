import os
import glob
import sys
import dlib

face_path = 'ibug_300W_large_face_landmark_dataset'

print(face_path)

# options = dlib.simple_object_detector_training_options()

# options.add_left_right_image_flips = True

# options.C = 5

# options.num_threads = 4

# options.be_verbose = True

# training_xml_path = os.path.join(face_path, 'face_mask_training_from_root.xml')
# testing_xml_path = os.path.join(face_path, 'face_mask_testing_from_root.xml')

# dlib.train_simple_object_detector(training_xml_path, "/Users/phuocvinh143/Git/Scientific-Project/model/detector.svm", options)

# print("")
# print('Training accuracy: {}'.format(dlib.test_simple_object_detector(training_xml_path, '/Users/phuocvinh143/Git/Scientific-Project-CTU/model/detector.svm')))
# print('Testing accuracy: {}'.format(dlib.test_simple_object_detector(testing_xml_path, '/Users/phuocvinh143/Git/Scientific-Project-CTU/model/detector.svm')))