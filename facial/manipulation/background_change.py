# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config

BACKGROUND_IMG_NAME = 'background.jpg'
BACKGROUND_IMG = os.path.join(
    config.basedir, 'static', 'img', BACKGROUND_IMG_NAME)
EROSION_FACTOR = 3


def initialize_mediapipe():
    """
    Initializing mediapipe selfie segmentation sub-module
    """
    mpSelfieSegmentation = mediapipe.solutions.selfie_segmentation.SelfieSegmentation(
        model_selection=1)

    return mpSelfieSegmentation


def refine_mask(img, erosion_factor):
    """
    Smooth the mask
    """
    out = img.copy()

    # Erode the mask to smooth boundaries
    # The pixel near the boundaries will be discarded depending on the size of the kernel
    # The thickness or size of the white area within the mask decreases
    size = erosion_factor
    # Defining the structuring element
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ERODE, (2*size+1, 2*size+1), (size, size))
    # BORDER_REFLECT - Depicts what kind of border
    out = cv2.erode(out, kernel, cv2.BORDER_REFLECT)
    return out


def change_background(input_path: str, display_output: bool = False):
    """
    Change the background
    """
    # Initialize mediapipe selfie segmentation sub-module
    mpSelfieSegmentation = initialize_mediapipe()

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()

    # Convert the image from BGR to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = []
    output_info = []

    # Process the RGB frame with the MediaPipe Selfie Segmentation Module
    processed_frame = mpSelfieSegmentation.process(rgb_frame)
    # Extract the mask out of the processed frame
    processed_mask = processed_frame.segmentation_mask
    processed_mask = refine_mask(processed_mask, erosion_factor=EROSION_FACTOR)

    # Return a matrix having the same shape as the mask.
    # The matrix contains True if pixel value > 0.2 and False if pixel value < 0.2
    overlay_condition = np.stack((processed_mask,)*3, axis=-1) > 0.2

    # Load the preset background frame
    background_frame = cv2.imread(BACKGROUND_IMG)
    # Resize the background frame to the same size of the selected image
    background_frame = cv2.resize(
        background_frame, (frame.shape[1], frame.shape[0]))

    # Combine both images where the condition is satisfied
    frame = np.where(overlay_condition, frame, background_frame)

    label = f"Changing Background to {BACKGROUND_IMG_NAME}"
    if display_output:
        # Display Image on screen
        cv2.imshow(label, frame)
        # Mantain output until user presses a key
        cv2.waitKey(0)
        # Cleanup
        cv2.destroyAllWindows()

    # Save and Output the resulting image
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex)+os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, frame)
    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER,
                   'name': os.path.basename(output_filepath), 'msg': label}
    output.append(output_item)

    mpSelfieSegmentation.close()
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
    change_background(
        input_path=args['input_path'], display_output=args['display_output'])
