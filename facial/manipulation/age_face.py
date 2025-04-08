# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
from tensorflow import keras
import config

# To reduce Mediapipe false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5
# GAN Model
AGING_GAN_MODEL = os.path.join(config.MODELS_PATH, 'generator.h5')
AGE_RANGE = 5

EXPANDING_FACTOR = 0.40


def initialize_mediapipe():
    """
    Initializing mediapipe face detection sub-module
    """
    # Enable face detection
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    return mpFaceDetection


def process_img(input_img):
    """
    Aging an input face image
    """
    # Load the model
    model = keras.models.load_model(AGING_GAN_MODEL)
    # print(model.summary())
    inputs = keras.Input((None, None, 4))
    outputs = model(inputs)

    model = keras.models.Model(inputs, outputs)

    orgH, orgW, orgC = input_img.shape

    img = cv2.resize(input_img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 255) * 2 - 1
    h, w, _ = img.shape

    # Selecting an Age Range
    condition = np.ones(shape=[1, h, w, 1]) * (AGE_RANGE - 1)
    conditioned_images = np.concatenate(
        [np.expand_dims(img, axis=0), condition], axis=-1)
    aged_img = model.predict(conditioned_images)[0]
    aged_img = (aged_img + 1.0) / 2.0
    aged_img = cv2.cvtColor(aged_img, cv2.COLOR_RGB2BGR)
    # Resize image according to original image
    aged_img = cv2.resize(aged_img, (orgW, orgH))
    # Convert from float32 to uint8
    aged_img = cv2.normalize(aged_img, None, 0, 255,
                             cv2.NORM_MINMAX, cv2.CV_8U)
    return aged_img


def refine_img(src_img):
    # Image denoising
    out = cv2.fastNlMeansDenoisingColored(
        # Source Image
        src=src_img                                    # Output Image
        , dst=None        # Filter strength regulator
        # Bigger value removes noise but also removes image details
        , h=5, hColor=10, templateWindowSize=7, searchWindowSize=21
    )
    # cv2.imshow('Denoised image',out)
    # cv2.waitKey(0)

    # Adjust brightness and contrast
    alpha = 1.0  # Contrast control (1.0 -> 3.0)
    beta = 0  # Brightness control (0 -> 100)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    # cv2.imshow('Adjusted image',out)
    # cv2.waitKey(0)
    return out


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


def aging_faces(input_path: str, display_output: bool = False):
    """
    Aging the faces detected within an image
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
        x, y, w, h = enlarge_bounding_box(x, y, w, h)

        # Extract the face -> region of interest
        roi_face_color = frame[y:y + h, x:x + w]

        # Process the face area
        aged_img = process_img(roi_face_color)
        frame[y:y + h, x:x + w] = aged_img

        # frame = refine_img(frame)

    if display_output:
        # Display Image on screen
        label = "Aging Faces"
        cv2.imshow(label, frame)
        # Mantain output until user presses a key
        cv2.waitKey(0)
        # Cleanup
        cv2.destroyAllWindows()
    # Save and Output the resulting image
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex)+os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, frame)
    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
        output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)
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
    aging_faces(input_path=args['input_path'],
                display_output=args['display_output'])
