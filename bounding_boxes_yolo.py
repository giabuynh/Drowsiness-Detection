from calculator_functions import *
from imutils import face_utils
import dlib
import imutils
import cv2
import os


def load_images_from_folder(folder_path):
    print('[INFO] Loading images from ', folder_path)
    filenames = os.listdir(folder_path)
    allowed_extension = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    images = []
    for filename in filenames:
        # get a full file path
        file_path = os.path.join(folder_path, filename)
        # check whether a file path is a file or a directory
        if os.path.isfile(file_path):
            # check whether a file is image file
            # try:
            # 	image = Image.open(file_path)
            # 	if image is not None:
            # 		images.append(filename)
            # 	image.close()
            # except:
            # 	continue
            filename_patterns = filename.split('.')
            if filename_patterns[-1] in allowed_extension:
                txt_filename = filename_patterns[0] + '.txt'
                if not txt_filename in filenames:
                    images.append(filename)
    return images


def write_txt_files_from_dict(save_folder_path, info_dict):
    for file in info_dict:
        filename = file.split('.')[0]

        txt_file = open(save_folder_path + filename + '.txt', 'w')
        txt_file.write(info_dict[file])
        txt_file.close()


def write_txt_file(save_folder_path, image_filename, content):
    filename = image_filename.split('.')[0]
    print('	[WRITE]', filename + '.txt')

    txt_file = open(save_folder_path + filename + '.txt', 'w')
    txt_file.write(content)
    txt_file.close()


def detect_bounding_boxes_by_dlib_yolo_format(image_set_path, label):
    image_set = load_images_from_folder(image_set_path)
    cnt = 0

    print('[INFO] Load detector...')
    detector = dlib.get_frontal_face_detector()
    print('[INFO] Detecting bounding boxes...')

    for image_filename in image_set:
        file_path = os.path.join(image_set_path, image_filename)
        cnt += 1

        print('	[INFO] Open', image_filename + '. (' + str(cnt) + '/' + str(len(image_set)) + ')')
        image = cv2.imread(file_path)
        (image_h, image_w) = image.shape[:2]

        try:
            faces = detector(image, 0)
            annotations = ''

            for face in faces:
                (x, y, w, h) = face_utils.rect_to_bb(face)
                x = x + w / 2
                y = y + w / 2

                x /= image_w
                y /= image_h
                w /= image_w
                h /= image_h

                if len(annotations) == 0:
                    annotations += label + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
                else:
                    annotations += '\n' + label + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)

            write_txt_file(image_set_path, image_filename, annotations)
        except:
            print('	[ERROR] Cannot detect image', image_filename, 'due to incompatible shape.')


image_set_path = 'D:/Thesis/MyData/IMFD/34000/'
detect_bounding_boxes_by_dlib_yolo_format(image_set_path, 'mask_weared_incorrect')
print('[COMPLETE] Detected bounding boxes')
