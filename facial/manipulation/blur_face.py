# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config

BLURRING_FACTOR = 3
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


def blur_img(input_img, blurring_factor):
    """
    Blurring the input image using the GAUSSIAN Blur function of opencv.
    This function takes 2 parameters:
    1) The input image
    2) The Gaussian kernel size (Kernel Size must be odd)
    Low blurring factors -> more blurred face
    """
    input_img_height, input_img_width = input_img.shape[:2]
    kWidth = int(input_img_width / blurring_factor)
    kHeight = int(input_img_height / blurring_factor)

    # Kernel shape must be odd
    if kWidth % 2 == 0:
        kWidth = kWidth + 1
    if kHeight % 2 == 0:
        kHeight = kHeight + 1
    blurred_img = cv2.GaussianBlur(
        src=input_img, ksize=(kWidth, kHeight), sigmaX=0)
    return blurred_img


def pixelate_img(input_img, blocks=3):
    # Divide the image into N*N blocks
    input_img_height, input_img_width = input_img.shape[:2]
    xSteps = np.linspace(0, input_img_width, blocks+1, dtype='int')
    ySteps = np.linspace(0, input_img_height, blocks+1, dtype='int')

    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX, startY = xSteps[j - 1], ySteps[i - 1]
            endX, endY = xSteps[j], ySteps[i]

            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = input_img[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(input_img, (startX, startY),
                          (endX, endY), (B, G, R), -1)

    # return the pixelated blurred image
    return input_img


def blur_faces(input_path: str, display_output: bool = False):
    """
    Blur the faces located within a digital image.
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

        # Extract the face -> region of interest
        roi_face_color = frame[y:y + h, x:x + w]

        # Blur the face area
        blurred_face = blur_img(roi_face_color, BLURRING_FACTOR)
        # blurred_face   = pixelate_img(roi_face_color)

        # Store the blurred face in the output image
        frame[y:y + h, x:x + w] = blurred_face

    if display_output:
        # Display Image on screen
        label = "Blurring Faces"
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
    blur_faces(input_path=args['input_path'],
               display_output=args['display_output'])
