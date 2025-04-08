# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
from deepface import DeepFace
import numpy as np
import config

EXPANDING_FACTOR = 0.25
# To reduce Mediapipe false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5


def initialize_mediapipe():
    """
    Initializing mediapipe face detection sub-module
    """
    # Enable face detection
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    return mpFaceDetection


def calculate_optimal_fontscale(text, width):
    """
    Determine the optimal font scale based on the hosting frame width
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        # print(new_width)
        if (new_width <= width):
            return scale/15
    return 1


def display_image(title, img):
    """
    Displays an image on screen and maintains the output until the user presses a key
    """
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setWindowTitle('img', title)
    cv2.resizeWindow('img', 600, 400)

    # Display Image on screen
    cv2.imshow('img', img)

    # Mantain output until user presses a key
    cv2.waitKey(0)

    # Destroy windows when user presses a key
    cv2.destroyAllWindows()


def enlarge_bounding_box(x, y, w, h):
    """
    Enlarge the bounding box based on the expanding factor
    """
    # create a larger bounding box with buffer around keypoints
    x1 = int(x - EXPANDING_FACTOR * w)
    w1 = int(w + 2 * EXPANDING_FACTOR * w)
    y1 = int(y - EXPANDING_FACTOR * h)
    h1 = int(h + 2 * EXPANDING_FACTOR * h)
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    print('x1,y1,w1,h1', x1, y1, w1, h1)
    return x1, y1, w1, h1


def predict_age_deepface(input_path: str, display_output: bool = False):
    """
    Predict the age of a face image
    """
    # Initialize mediapipe face detection sub-module
    mpFaceDetection = initialize_mediapipe()

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()

    # Convert the image from BGR to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using the rgb frame
    faces = mpFaceDetection.process(rgb_frame)

    output = []
    output_info = []

    # Output message the number of faces detected...
    output_msg = {'msg': "{} face(s) detected.".format(
        len(faces.detections)), 'category': "info"}
    output_info.append(output_msg)

    # Loop over the faces detected
    for idx, face_detected in enumerate(faces.detections):

        output_item = None
        label = f"Face ID = {(idx+1)} - Detection Score {int(face_detected.score[0]*100)}%"
        print(label)

        # Get the relative bounding box
        relativeBoundingBox = face_detected.location_data.relative_bounding_box
        frameHeight, frameWidth, frameChannels = frame.shape
        faceBoundingBox = int(relativeBoundingBox.xmin*frameWidth), int(relativeBoundingBox.ymin*frameHeight), int(
            relativeBoundingBox.width*frameWidth), int(relativeBoundingBox.height*frameHeight)

        # Get the coordinates of the face bounding box
        x, y, w, h = faceBoundingBox

        # Create a larger bounding box with buffer around keypoints
        x1, y1, w1, h1 = enlarge_bounding_box(x, y, w, h)

        # Crop the enlarged region for better analysis
        roi_face_color = frame[y1:y1+h1, x1:x1+w1]

        try:
            # Analyze the croped image using deepface
            result = DeepFace.analyze(
                np.array(roi_face_color), actions=['age'])
            print('### DeepFace.analyze result = ', result)

            if result['age']:
                # Draw the box
                label = "AGE {}".format(result['age'])
                print(label)

                # Calculate the optimal font scale for the output label
                optimal_font_scale = calculate_optimal_fontscale(
                    label, (((x + w) - x)))

                # Draw the output label
                optimal_font_scale = 1 if optimal_font_scale < 1 else optimal_font_scale
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (0, 255, 0), 2 * round(optimal_font_scale))
                labelSize = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_DUPLEX, round(optimal_font_scale), 2)
                _x1 = x
                _y1 = y+h
                _x2 = _x1 + labelSize[0][0]
                _y2 = _y1 + int(labelSize[0][1])
                cv2.rectangle(frame, (_x1, _y1), (_x2, _y2),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (_x1, _y2), cv2.FONT_HERSHEY_DUPLEX, int(
                    optimal_font_scale), (0, 0, 0), 2)
            else:
                # Draw a red rectangle around faces with unpredicted age
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        except Exception as err:
            print('Exception Found', Exception, err)
            # Draw a red rectangle around faces in case of exception
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if display_output:
            # Display Image on screen
            label = "Age Estimator Using DeepFace"
            cv2.imshow(label, frame)
            # Mantain output until user presses a key
            cv2.waitKey(0)

    if display_output:
        # Cleanup
        cv2.destroyAllWindows()

    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex) + os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, frame)

    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
        output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)
    mpFaceDetection.close()
    return output_info, output


def is_valid_path(path):
    """
    Validates the path inputted and validates that it is a file of type image
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
    predict_age_deepface(
        input_path=args['input_path'], display_output=args['display_output'])
