# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config
import imutils

FRAME_WIDTH = 500
FRAME_HEIGHT = 500

# To reduce false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5


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

#############################################################################################################


def grab_roi(face_coordinates, roi_coordinates):
    """
    Grab the coordinates corresponding to the region of interest.
    """
    area_x = []
    area_y = []
    for i in roi_coordinates:
        area_x.append(face_coordinates[i][0])
        area_y.append(face_coordinates[i][1])

    x_min = min(area_x)
    x_max = max(area_x)
    y_min = min(area_y)
    y_max = max(area_y)

    return x_min, x_max, y_min, y_max


def detect_nose_area(img, face_coordinates):
    """
    Checking the nose area
    """
    result = 0
    # Nose Area
    x_min, x_max, y_min, y_max = grab_roi(face_coordinates, roi_coordinates=[
                                          9, 8, 168, 6, 197, 195, 5, 4, 1])

    considered_area = [x_min, y_min, x_max, y_max]
    roi = img[y_min:y_max, x_min:x_max]

    roi_blur = cv2.GaussianBlur(roi, (3, 3), sigmaX=0, sigmaY=0)
    roi_edges = cv2.Canny(image=roi_blur, threshold1=100, threshold2=200)

    if 255 in roi_edges:  # roi_edges_center:
        result = 1

    return result, considered_area


def detect_lefteye_area(img, face_coordinates):
    """
    Checking the left eye area
    """
    result = 0
    # Left Eye Area
    x_min, x_max, y_min, y_max = grab_roi(
        face_coordinates, roi_coordinates=[229, 114, 50, 131])

    considered_area = [x_min, y_min, x_max, y_max]
    roi = img[y_min:y_max, x_min:x_max]

    roi_blur = cv2.GaussianBlur(roi, (3, 3), sigmaX=0, sigmaY=0)
    roi_edges = cv2.Canny(image=roi_blur, threshold1=100, threshold2=200)
    roi_edges_center = roi_edges.T[(int(len(roi_edges.T) / 2))]
    if 255 in roi_edges_center:  # roi_edges:
        result = 1
    return result, considered_area


def detect_righteye_area(img, face_coordinates):
    """
    Checking the right eye area
    """
    result = 0
    # Right Eye Area
    x_min, x_max, y_min, y_max = grab_roi(
        face_coordinates, roi_coordinates=[343, 448, 429, 280])
    considered_area = [x_min, y_min, x_max, y_max]
    roi = img[y_min:y_max, x_min:x_max]

    roi_blur = cv2.GaussianBlur(roi, (3, 3), sigmaX=0, sigmaY=0)
    roi_edges = cv2.Canny(image=roi_blur, threshold1=100, threshold2=200)
    roi_edges_center = roi_edges.T[(int(len(roi_edges.T) / 2))]
    if 255 in roi_edges_center:  # roi_edges:
        result = 1
    return result, considered_area


def grab_face_coordinates(img, img_landmarks):
    """
    Grab the coordinates of the face landmarks
    """
    face_coordinates = np.array([[int(min([landmark.x * img.shape[1], img.shape[1] - 1])), int(min([landmark.y * img.shape[0], img.shape[0] - 1]))]
                                 for landmark in img_landmarks.landmark]).astype(int)
    face_coordinates[face_coordinates < 0] = 0
    return face_coordinates


def detect_eyeglasses(input_path: str, display_output: bool = False):
    """
    Detect faces within a digital image and check for eyeglasses.
    """
    # Initialize the mediapipe sub-modules
    mpFaceDetection, mpFaceModule = initialize_mediapipe()

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()
    initial_frame = img.copy()

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

        # At each iteration take a new frame copy
        frame = img.copy()

        # Determine the face bounding box
        relativeBoundingBox = face_detected.location_data.relative_bounding_box
        frameHeight, frameWidth, frameChannels = frame.shape
        faceBoundingBox = int(relativeBoundingBox.xmin*frameWidth), int(relativeBoundingBox.ymin*frameHeight), int(
            relativeBoundingBox.width*frameWidth), int(relativeBoundingBox.height*frameHeight)

        # Draws a rectangle over the face bounding box
        cv2.rectangle(frame, faceBoundingBox, (0, 255, 0), 3)
        x, y, w, h = faceBoundingBox
        startX, startY, endX, endY = x, y, x+w, y+h

        # Region of interest in color mode
        roi_face_color = frame[y:y+h, x:x+w]

        # Grab the coordinate of the face landmark points
        face_coordinates = grab_face_coordinates(frame, face_landmarks)

        result_nose, area_nose = detect_nose_area(
            img=frame, face_coordinates=face_coordinates)
        result_leye, area_leye = detect_lefteye_area(
            img=frame, face_coordinates=face_coordinates)
        result_reye, area_reye = detect_righteye_area(
            img=frame, face_coordinates=face_coordinates)
        print('Results --Nose = ', result_nose, '--Left Eye=',
              result_leye, '--Right Eye=', result_reye)

        cv2.rectangle(
            initial_frame, (area_nose[0], area_nose[1]), (area_nose[2], area_nose[3]), (0, 255, 0), 2)
        cv2.rectangle(
            initial_frame, (area_leye[0], area_leye[1]), (area_leye[2], area_leye[3]), (0, 255, 0), 2)
        cv2.rectangle(
            initial_frame, (area_reye[0], area_reye[1]), (area_reye[2], area_reye[3]), (0, 255, 0), 2)

        label = ""
        if result_nose == 1 and (result_leye == 1 or result_reye == 1):
            initial_frame = cv2.rectangle(
                initial_frame, (startX, startY), (endX, endY), (255, 0, 0), 5)
            cv2.putText(initial_frame, 'Detected', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
            label = 'Eyeglasses Detected...'
        elif (result_leye == 1 and result_reye == 1):
            cv2.rectangle(initial_frame, (startX, startY),
                          (endX, endY), (255, 0, 0), 5)
            cv2.putText(initial_frame, 'Detected', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
            label = 'Eyeglasses Detected...'
        else:
            cv2.rectangle(initial_frame, (startX, startY),
                          (endX, endY), (0, 0, 255), 5)
            cv2.putText(initial_frame, 'Not Detected', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            label = 'Eyeglasses Not Detected...'
        print(label)
######################
    output_filepath = os.path.join(config.PROCESSED_PATH,
                                   str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, initial_frame)
    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
        output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)
######################
    if display_output:
        # Display Image on screen
        label = "Eyeglasses Detector"
        frame = imutils.resize(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT)
        cv2.imshow(label, initial_frame)
        # Maintain output until user presses a key
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
    detect_eyeglasses(
        input_path=args['input_path'], display_output=args['display_output'])
