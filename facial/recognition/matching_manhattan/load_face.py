# Import Libraries
import os
import argparse
import uuid
import cv2
import filetype
import imutils
from faceRecognitionTools import FaceRecognitionTools
import config
from PIL import Image
from PIL import ImageDraw
import numpy as np

# Frame Resize
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
# Face detection model to use
# hog --> less accurate but faster on CPUs.
# cnn --> more accurate deep-learning model which is GPU/CUDA accelerated (if available).
# (Default is hog)
MODEL = "hog"


def draw_landmarks(frame, face_landmarks):
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # Loop over the facial features
    for name, list_of_points in face_landmarks.items():
        # Trace out each facial feature
        draw.line(list_of_points, fill='green', width=3)

    cv_img = np.array(pil_img)
    # cv_img = cv_img[:,:,::-1].copy()
    return cv_img


def load_faces(input_path: str, display_output: bool = False):
    """
    Save detected faces in a digital image to database
    """
    output = []
    output_info = []

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()
    frame = imutils.resize(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT)

    face_recognition_tools = FaceRecognitionTools()

    for idx, face_location, face_encoding, face_landmarks \
            in face_recognition_tools.grab_faces_and_landmarks(img=frame, model=MODEL):
        # print(idx , face_location, face_encoding)
        (top, right, bottom, left) = face_location
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        # Region of interest in color mode
        roi_face_color = frame[top:bottom, left:right]

        dbFaceID = face_recognition_tools.add_Face(img=frame, img_path=input_path, face_ref=idx, face_location=str(face_location), face_encoding=face_encoding, searchFor=False
                                                   )
        label = f"Face loaded to Database - ID = {dbFaceID}"
        print(label)
        ##########################################################
        output_filepath = os.path.join(config.PROCESSED_PATH, str(
            uuid.uuid4().hex)+os.path.splitext(input_path)[1])
        cv2.imwrite(output_filepath, roi_face_color)
        output_item = {'id': idx, 'folder': config.PROCESSED_FOLDER,
                       'name': os.path.basename(output_filepath), 'msg': label}
        output.append(output_item)
        ##########################################################
        fl_img = draw_landmarks(frame, face_landmarks)
        output_filepath = os.path.join(config.PROCESSED_PATH, str(
            uuid.uuid4().hex)+os.path.splitext(input_path)[1])
        cv2.imwrite(output_filepath, fl_img)
        output_item = {'id': idx, 'folder': config.PROCESSED_FOLDER,
                       'name': os.path.basename(output_filepath), 'msg': label}
        output.append(output_item)
        ##########################################################
        if display_output:
            # Display Image on screen
            cv2.imshow(label, roi_face_color)
            # Mantain output until user presses a key
            cv2.waitKey(0)

            cv2.imshow(label, fl_img)
            # Mantain output until user presses a key
            cv2.waitKey(0)

    if display_output:
        # Cleanup
        cv2.destroyAllWindows()

    return output_info, output


def is_valid_path(path):
    """
    Validates the path inputted and makes sure that is a file of type image
    """
    if not path:
        raise ValueError(f"Invalid Path")
    if os.path.isfile(path) and 'image' in filetype.guess(path).mime:
        return path
    else:
        raise ValueError(f"Invalid Path {path}")


def parse_args():
    """
    Get user command line parameters
    """
    parser = argparse.ArgumentParser(description="Available Options")

    parser.add_argument('-i', '--input_path', dest='input_path', type=is_valid_path,
                        required=True, help="Enter the path of the image file to process")

    parser.add_argument('-d', '--display_output', dest='display_output', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help="Display output on screen")

    args = vars(parser.parse_args())

    # To Display The Command Line Arguments
    print("## Command Arguments #################################################")
    print("\n".join("{}:{}".format(i, j) for i, j in args.items()))
    print("######################################################################")

    return args


if __name__ == '__main__':
    # Parsing command line arguments entered by user
    args = parse_args()
    load_faces(input_path=args['input_path'],
               display_output=args['display_output'])
