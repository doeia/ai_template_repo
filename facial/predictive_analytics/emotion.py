# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config

# To reduce Mediapipe false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5

EXPANDING_FACTOR = 0.25


def initialize_mediapipe():
    """
    Initializing mediapipe face detection sub-module
    """
    # Enable face detection
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    return mpFaceDetection


def layout_emotions(emotions):
    """
    Layout the emotions in an image
    """
    emotions_img = Image.new('RGB', (300, 300), color=(0, 0, 0, 0))
    # Get a drawing context
    draw = ImageDraw.Draw(emotions_img)
    font = ImageFont.truetype(os.path.join(
        config.FONTS_PATH, 'OpenSansEmoji.ttf'), 30, encoding='unic')
    emotions = sorted(emotions.items(), key=lambda tup: tup[1], reverse=True)
    for idx, (emotion, score) in enumerate(emotions):
        if emotion:
            score = 0 if score is None else score
            label = "{}-{:.2f}%".format(emotion, score * 100)
            draw.text((20, 40 * idx), label, font=font, embedded_color=True)

    emotions_img = np.array(emotions_img)
    return emotions_img


def get_emotion(img):
    """
    Gather emotions
    """
    from fer import FER
    detector = FER(mtcnn=True)
    result = detector.detect_emotions(img)
    emotion, emotion_score = detector.top_emotion(img)
    emotion_score = 0 if emotion_score is None else emotion_score
    top_emotion = "Top emotion: {} - {:.2f}%".format(
        emotion, emotion_score * 100)
    print(top_emotion)
    emotions_layout = None
    if result:
        emotions = result[0]["emotions"]
        emotions_layout = layout_emotions(emotions)
    return top_emotion, emotions_layout


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


def predict_emotions(input_path: str, display_output: bool = False):
    """
    Predict the emotions of the faces showing in the image
    """
    # Initialize mediapipe face detection sub-module
    mpFaceDetection = initialize_mediapipe()

    # Read Input Image
    img = cv2.imread(input_path)

    # Copy the initial image
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
        label = f"Face ID = {(idx+1)} - Detection Score {int(face_detected.score[0]*100)}%"
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

        # Determine the region of interest
        roi = frame[y1:y1+h1, x1:x1+w1]

        # Get the corresponsing emotion
        top_emotion, emotions_layout = get_emotion(img=roi)
        label = f"Face ID = {(idx + 1)} - Detection Score {int(face_detected.score[0] * 100)}% - {top_emotion}"

        # Concatenate the emotions layout with the face image
        combined_img = cv2.resize(roi, (400, 400), cv2.INTER_CUBIC)
        if emotions_layout is not None:
            combined_img = cv2.hconcat([cv2.resize(roi, (400, 400), cv2.INTER_CUBIC), cv2.resize(
                emotions_layout, (400, 400), cv2.INTER_CUBIC)])

            # Output the combined image
            output_filepath = os.path.join(config.PROCESSED_PATH,
                                           str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
            cv2.imwrite(output_filepath, combined_img)

            output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER,
                           'name': os.path.basename(output_filepath), 'msg': label}
            output.append(output_item)

        if display_output:
            # Display Image on screen
            cv2.imshow(top_emotion, combined_img)
            # Mantain output until user presses a key
            cv2.waitKey(0)

    if display_output:
        # Cleanup
        cv2.destroyAllWindows()

    mpFaceDetection.close()

    return output_info, output


def is_valid_path(path):
    """
    Validates the path inputted and makes sure that it is a file of type image
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
    predict_emotions(input_path=args['input_path'],
                     display_output=args['display_output'])
