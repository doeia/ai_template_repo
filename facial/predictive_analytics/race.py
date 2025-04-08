# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
from deepface import DeepFace
import config
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# To reduce Mediapipe false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5

EXPANDING_FACTOR = 0.4


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


def layout_races(races):
    """
    Layout list of races in an image
    """
    races_img = Image.new('RGB', (300, 300), color=(0, 0, 0, 0))
    # Get a drawing context
    draw = ImageDraw.Draw(races_img)
    font = ImageFont.truetype(os.path.join(
        config.FONTS_PATH, 'OpenSansEmoji.ttf'), 26, encoding='unic')
    races = sorted(races.items(), key=lambda kv: kv[1], reverse=True)

    for idx, (race, score) in enumerate(races):
        if race:
            score = 0 if score is None else score
            label = "{}-{:.0f}%".format(race, round(score))
            draw.text((20, 40 * idx), label, font=font, embedded_color=True)

    races_img = np.array(races_img)
    return races_img


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


def predict_race_deepface(input_path: str, display_output: bool = False):
    """
    Predict the races of the faces showing in the image
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
        # Output message
        label = f"Face ID = {(idx + 1)} - Detection Score {int(face_detected.score[0] * 100)}%"
        output_msg = {'msg': label, 'category': "info"}
        output_info.append(output_msg)
        print(output_msg.get('category'), output_msg.get('msg'))

        # Get the face relative bounding box
        relativeBoundingBox = face_detected.location_data.relative_bounding_box
        frameHeight, frameWidth, frameChannels = frame.shape
        faceBoundingBox = int(relativeBoundingBox.xmin * frameWidth), int(relativeBoundingBox.ymin * frameHeight), int(
            relativeBoundingBox.width * frameWidth), int(relativeBoundingBox.height * frameHeight)
        # Get the coordinates of the face bounding box
        x, y, w, h = faceBoundingBox

        # Enlarge the bounding box
        x1, y1, w1, h1 = enlarge_bounding_box(x, y, w, h)

        roi_face_color = frame[y1:y1+h1, x1:x1+w1]

        try:
            result = DeepFace.analyze(
                np.array(roi_face_color), actions=['race'])

            if result:
                # Draw the box
                label = f"Face ID={(idx+1)} - Detection Score={int(face_detected.score[0]*100)}% - Top Race={result['dominant_race']}"
                print(label)

                races_layout = layout_races(result['race'])
                combined_img = cv2.resize(frame, (400, 400), cv2.INTER_CUBIC)

                roi_face_color = frame[y:y + h, x:x + w]
                if races_layout is not None:
                    combined_img = cv2.hconcat([cv2.resize(roi_face_color, (400, 400), cv2.INTER_CUBIC), cv2.resize(
                        races_layout, (400, 400), cv2.INTER_CUBIC)])
        except:
            # Draw a red rectangle around faces with unpredicted race
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
            combined_img = frame

        output_filepath = os.path.join(config.PROCESSED_PATH,
                                       str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
        cv2.imwrite(output_filepath, combined_img)

        output_item = {'id': (idx + 1), 'folder': config.PROCESSED_FOLDER,
                       'name': os.path.basename(output_filepath), 'msg': label}
        output.append(output_item)

        if display_output:
            # Display Image on screen
            cv2.imshow(result['dominant_race'], combined_img)
            # Mantain output until user presses a key
            cv2.waitKey(0)

    if display_output:
        # Cleanup
        cv2.destroyAllWindows()

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
    predict_race_deepface(
        input_path=args['input_path'], display_output=args['display_output'])
