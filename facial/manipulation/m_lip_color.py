# Import Libraries
import os
import argparse
import uuid
import dlib
import cv2
import filetype
import numpy as np
from imutils import face_utils
import config

# Landmark's facial detector to estimate the location of 68 coordinates that map the facial points
# in a person's face
FACIAL_LANDMARK_PREDICTOR = os.path.join(
    config.MODELS_PATH, 'shape_predictor_68_face_landmarks.dat')

# Targeted Color
TARGET_COLOR = (0, 0, 0)


def initialize_dlib(facial_landmark_predictor: str):
    """
    Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    """
    print('Loading facial landmark predictor...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

    return detector, predictor


def change_lipstick_color(shape, img, mouth_landmark):
    """
    Change the lipstick color
    """
    name, (i, j) = mouth_landmark
    points = []
    for (x, y) in shape[i:j]:
        points.append([x, y])
    # Convert the list of points into a numpy array of suitable dimension
    np_points = np.reshape(points, (-1, 1, 2))
    # Draw a filled polygon on the points selected
    cv2.fillPoly(img, [np_points], TARGET_COLOR, 8)
    return img


def change_lips_color(input_path: str, display_output: bool = False):
    """
    Change the lipstick color of the faces within a digital image.
    """
    # Initialize dlib face detector using the facial landmark recognition
    detector, predictor = initialize_dlib(
        facial_landmark_predictor=FACIAL_LANDMARK_PREDICTOR)

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()

    # Convert it to gray scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray scale frame
    faces = detector(gray_frame, 0)

    output = []
    output_info = []

    # Loop over the faces detected
    for idx, face in enumerate(faces):
        output_msg = {'msg': "Face {} detected on position (Left:{} Top:{} Right:{} Botton:{}).".
                      format((idx+1), face.left(), face.top(), face.right(), face.bottom()), 'category': "info"}
        output_info.append(output_msg)
        print(output_msg.get('category'), output_msg.get('msg'))

        # Determine the facial landmarks for the face region
        # Convert the facial landmarks to a Numpy Arroy
        shape = predictor(gray_frame, face)
        shape = face_utils.shape_to_np(shape)

        # List containing the facial features
        face_landmarks = list(face_utils.FACIAL_LANDMARKS_IDXS.items())

        # The 8 landmarks of the face
        # print('len(face_landmarks) = ',len(face_landmarks))
        # Facial features were detected
        if (len(face_landmarks)):
            # The mouth landmark
            # print('face_landmarks[0]',face_landmarks[0])
            # Represents the mouth face_landmarks[0]
            change_lipstick_color(shape, frame, face_landmarks[0])

        if display_output:
            # Display Image on screen
            label = "Facial Landmarks"
            cv2.imshow(label, frame)
            # Mantain output until user presses a key
            cv2.waitKey(0)
            # Cleanup
            cv2.destroyAllWindows()

    output_filepath = os.path.join(config.PROCESSED_PATH, str(uuid.uuid4().hex)
                                   + os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, frame)
    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
        output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)
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
    change_lips_color(
        input_path=args['input_path'], display_output=args['display_output'])
