# Import Libraries
import os
import argparse
import cv2
import filetype
import imutils
from faceRecognitionTools import FaceRecognitionTools
import config

# Frame Resize
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
# Face detection model to use
# hog --> less accurate but faster on CPUs.
# cnn --> more accurate deep-learning model which is GPU/CUDA accelerated (if available).
# (Default is hog)
MODEL = "hog"
DISTANCE = 5


def find_faces_manhattan_distance(input_path: str, display_output: bool = False):
    """
    Find face in the database of loaded faces
    """
    output = []
    output_info = []

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()
    # Resize image
    frame = imutils.resize(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    #
    face_recognition_tools = FaceRecognitionTools()

    for idx, face_location, face_encoding in face_recognition_tools.grab_Faces(img=frame, model=MODEL):
        (top, right, bottom, left) = face_location
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        # Region of interest in color mode
        roi_face_color = frame[top:bottom, left:right]

        dbFaceID = face_recognition_tools.add_Face(img=frame, img_path=input_path, face_ref=idx, face_location=str(face_location), face_encoding=face_encoding, searchFor=True
                                                   )
        label = f"Face loaded to Database Searchable Table - ID = {dbFaceID}"
        print(label)

        similarFaces = face_recognition_tools.searchSimilarFaces_Manhattan(
            dbFaceID, DISTANCE)
        # print("results=", similarFaces)

        for idx, similarFace in enumerate(similarFaces):
            # print(similarFace[2],similarFace[3],similarFace[4])

            label = f"Face Detected FaceID={similarFace[2]} - Distance={similarFace[4]:.2f}"
            print(label)

            output_filepath = os.path.join(config.UPLOAD_PATH, similarFace[3])
            output_item = {'id': idx, 'folder': config.UPLOAD_FOLDER,
                           'name': similarFace[3], 'msg': label}
            output.append(output_item)

            if display_output:
                similar_img = cv2.imread(output_filepath)
                # Display Image on screen
                cv2.imshow(label, similar_img)
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
    find_faces_manhattan_distance(
        input_path=args['input_path'], display_output=args['display_output'])
