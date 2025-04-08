# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config

# To reduce false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5
TARGET_COLOR = (0, 0, 0)
CONDENSE_FACTOR = 0.3


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


def enhance_image(img):
    # Contrast limited adaptive histogram equalization
    c = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    # Convert from BGR color space to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Split to 3 different channels
    l, a, b = cv2.split(lab)
    # Apply clache to the L channel
    l2 = c.apply(l)
    # Merge channels
    tmp = cv2.merge((l2, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def process_right_eyebrow(img, face_coordinates, color):
    """
    Process the right eyebrow based on the given face coordinates
    """
    # Right eyebrow landmark points
    r_eyebrow = []
    for pt in (300, 293, 334, 296, 336, 285, 295, 282, 283, 276):
        r_eyebrow.append(face_coordinates[pt])

    pts = np.array(r_eyebrow, np.int32)

    # cv2.polylines(img,[pts],False,color,thickness=0)
    cv2.fillPoly(img, [pts], color, 8)


def process_left_eyebrow(img, face_coordinates, color):
    """
    Process the left eyebrow based on the given face coordinates
    """
    # Right eyebrow landmark points
    l_eyebrow = []
    for pt in (70, 63, 105, 66, 107, 55, 65, 52, 53, 46):
        l_eyebrow.append(face_coordinates[pt])

    pts = np.array(l_eyebrow, np.int32)

    cv2.fillPoly(img, [pts], color, 8)


##############################################################################################
def grab_face_coordinates(img, img_landmarks):
    """
    Grab the coordinates of the face landmarks
    """
    face_coordinates = np.array([[int(min([landmark.x * img.shape[1], img.shape[1] - 1])), int(min([landmark.y * img.shape[0], img.shape[0] - 1]))]
                                 for landmark in img_landmarks.landmark]).astype(int)
    face_coordinates[face_coordinates < 0] = 0
    return face_coordinates


def condense_eyebrows(input_path: str, display_output: bool = False):
    """
    Condense the eyebrows of the faces showing within a digital image
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
        x, y, w, h = faceBoundingBox

        # Grab the coordinate of the face landmark points
        face_coordinates = grab_face_coordinates(frame, face_landmarks)
######################
        # At each iteration take a new frame copy
        frame_to_process = frame.copy()

        # Process Right Brow
        process_right_eyebrow(
            frame_to_process, face_coordinates, color=TARGET_COLOR)

        # Process Left Eye
        process_left_eyebrow(
            frame_to_process, face_coordinates, color=TARGET_COLOR)

        frame = cv2.addWeighted(
            frame_to_process, CONDENSE_FACTOR, frame, 1-CONDENSE_FACTOR, 0)
######################
    # Output the processed image
    output_filepath = os.path.join(config.PROCESSED_PATH,
                                   str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, frame)
    output_item = {'id': 3, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
        output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)
######################
    if display_output:
        # Display Image on screen
        label = "Eyebrows Condenser"
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
    condense_eyebrows(
        input_path=args['input_path'], display_output=args['display_output'])
