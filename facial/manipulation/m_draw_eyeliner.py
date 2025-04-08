# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config
from scipy import interpolate

# To reduce false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5

LINER_EXTENSION_POINTS = 1
LINER_COLOR = (0, 0, 0)


def initialize_mediapipe():
    """
    Initializing mediapipe sub-modules
    """
    # Enable the face detection sub-module
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    # Enable the face mesh sub-module
    mpFaceModule = mediapipe.solutions.face_mesh

    return mpFaceDetection, mpFaceModule
############################################################################################


def calculate_thickness_factor(frame):
    thickness_factor = (frame.shape[0] * frame.shape[1]) / \
        ((frame.shape[0] + frame.shape[1]) * 100)
    thickness_factor = 1 if thickness_factor < 1 else int(thickness_factor)
    return thickness_factor
##############################################################################################


def grab_face_coordinates(img, img_landmarks):
    """
    Grab the coordinates of the face landmarks
    """
    face_coordinates = np.array([[int(min([landmark.x * img.shape[1], img.shape[1] - 1])), int(min([landmark.y * img.shape[0], img.shape[0] - 1]))]
                                 for landmark in img_landmarks.landmark]).astype(int)
    face_coordinates[face_coordinates < 0] = 0
    return face_coordinates
##############################################################################################


def getLeftEyeLandmarkPts(face_coordinates):
    '''
    Get Left eye's landmark points
    '''
    face_coordinates[33][0] -= LINER_EXTENSION_POINTS
    face_coordinates[133][0] += LINER_EXTENSION_POINTS

    # Left Eye Top Area
    eye_top = [face_coordinates[i].tolist() for i in (
        33, 246, 161, 160, 159, 158, 157, 173, 133)]
    # Left Eye Bottom Area
    eye_bottom = [face_coordinates[i].tolist()
                  for i in (33, 7, 163, 144, 145, 153, 154, 155, 133)]

    return [np.asarray(eye_top), np.asarray(eye_bottom)]


def getRightEyeLandmarkPts(face_coordinates):
    '''
    Get Right eye's landmark points
    '''
    face_coordinates[362][0] -= LINER_EXTENSION_POINTS
    face_coordinates[263][0] += LINER_EXTENSION_POINTS

    # Right Eye Top Area
    eye_top = [face_coordinates[i].tolist() for i in (
        362, 398, 384, 385, 386, 387, 388, 466, 263)]
    # Right Eye Bottom Area
    eye_bottom = [face_coordinates[i].tolist()
                  for i in (362, 382, 381, 380, 374, 373, 390, 249, 263)]

    return [np.asarray(eye_top), np.asarray(eye_bottom)]


def interpolateCoordinates(point_coords, x_intrp):
    point_coords = point_coords.tolist()
    coordinates, x_coordinates = [], []
    for x in point_coords:
        if x[0] not in x_coordinates:
            coordinates.append(x)
            x_coordinates.append(x[0])
    unique_coordinates = np.array(coordinates)
    # print('x',unique_coordinates[:, 0])
    # print('y',unique_coordinates[:, 1])
    x = unique_coordinates[:, 0]
    y = unique_coordinates[:, 1]
    # print('x',x)
    # print('y',y)
    intrp = interpolate.interp1d(x, y, kind='quadratic')
    y_intrp = intrp(x_intrp)
    y_intrp = np.floor(y_intrp).astype(int)
    return y_intrp


def getEyelinerPoints(face_coordinates, eye_top, eye_bottom):
    '''
    Get eyeliner points
    '''
    # print('eye_top'   ,eye_top  )
    # print('eye_bottom',eye_bottom)

    interpolation_x = np.arange(eye_top[0][0], eye_top[-1][0], 1)
    # print('interpolation_x', interpolation_x)

    interpolation_top_y = interpolateCoordinates(eye_top, interpolation_x)
    interpolation_bottom_y = interpolateCoordinates(
        eye_bottom, interpolation_x)
    # print('interpolation_top_y'   ,interpolation_top_y)
    # print('interpolation_bottom_y',interpolation_bottom_y)

    return [(interpolation_x, interpolation_top_y, interpolation_bottom_y)]


def drawEyeliner(frame, interpolation_pts, color, thickness):
    [(interpolation_x, interpolation_top_y, interpolation_bottom_y)] = interpolation_pts

    processed_frame = frame.copy()

    for i in range(len(interpolation_x) - 2):
        x1 = interpolation_x[i]
        y1_top = interpolation_top_y[i]
        x2 = interpolation_x[i + 1]
        y2_top = interpolation_top_y[i + 1]
        cv2.line(processed_frame, (x1, y1_top), (x2, y2_top), color, thickness)

        y1_bottom = interpolation_bottom_y[i]
        y2_bottom = interpolation_bottom_y[i + 1]
        cv2.line(processed_frame, (x1, y1_bottom),
                 (x1, y2_bottom), color, thickness)

    return processed_frame


def draw_eyeliners(input_path: str, display_output: bool = False):
    """
    Draw eyeliners on faces showing within an image
    """
    # Initialize the mediapipe sub-modules
    mpFaceDetection, mpFaceModule = initialize_mediapipe()

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()

    # Convert it to rgb format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the rgb frame
    faces = mpFaceDetection.process(rgb_frame)

    output = []
    output_info = []

    # Output a message showing the faces detected
    output_msg = {'msg': "{} face(s) detected.".format(
        len(faces.detections)), 'category': "info"}
    output_info.append(output_msg)

    mpFaceMesh = mpFaceModule.FaceMesh(
        static_image_mode=True, max_num_faces=len(faces.detections), refine_landmarks=True, min_detection_confidence=MIN_CONFIDENCE_LEVEL
    )
    landmarks = mpFaceMesh.process(rgb_frame).multi_face_landmarks

    # Loop over the faces detected
    for idx, (face_detected, face_landmarks) in enumerate(zip(faces.detections, landmarks)):
        # Output message
        label = f"Face ID = {(idx + 1)} - Detection Score {int(face_detected.score[0] * 100)}%"
        output_msg = {'msg': label, 'category': "info"}
        output_info.append(output_msg)
        print(output_msg.get('category'), output_msg.get('msg'))

        # Determine the face bounding box
        relativeBoundingBox = face_detected.location_data.relative_bounding_box
        frameHeight, frameWidth, frameChannels = frame.shape
        faceBoundingBox = int(relativeBoundingBox.xmin*frameWidth), int(relativeBoundingBox.ymin*frameHeight), int(
            relativeBoundingBox.width*frameWidth), int(relativeBoundingBox.height*frameHeight)

        # Draws a rectangle over the face bounding box
        cv2.rectangle(frame, faceBoundingBox, (0, 255, 0), 3)
        # Grab the coordinate of the face landmark points
        face_coordinates = grab_face_coordinates(frame, face_landmarks)
######################
        thickness_factor = calculate_thickness_factor(frame)

        left_eye_top, left_eye_bottom = getLeftEyeLandmarkPts(face_coordinates)
        left_eye_liner_points = getEyelinerPoints(
            face_coordinates, left_eye_top, left_eye_bottom)

        frame = drawEyeliner(frame, left_eye_liner_points,
                             color=LINER_COLOR, thickness=thickness_factor)

        right_eye_top, right_eye_bottom = getRightEyeLandmarkPts(
            face_coordinates)
        right_eye_liner_points = getEyelinerPoints(
            face_coordinates, right_eye_top, right_eye_bottom)
        frame = drawEyeliner(frame, right_eye_liner_points,
                             color=LINER_COLOR, thickness=thickness_factor)
######################
    # Output the processed image
    output_filepath = os.path.join(config.PROCESSED_PATH,
                                   str(uuid.uuid4().hex)+os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, frame)
    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
        output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)
######################
    if display_output:
        # Display Image on screen
        label = "Drawing Eyeliners"
        cv2.imshow(label, frame)
        # Mantain output until user presses a key
        cv2.waitKey(0)
        # Cleanup
        cv2.destroyAllWindows()

    mpFaceMesh.close()
    mpFaceDetection.close()
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
    draw_eyeliners(input_path=args['input_path'],
                   display_output=args['display_output'])
