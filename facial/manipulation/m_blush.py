# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config
from scipy.spatial import distance

# To reduce false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5

# Rg, Gg, Bg = (223., 91., 111.)
Rg, Gg, Bg = (153., 0., 157.)

# Reduce this factor to intensify blush
BLUSH_INTENSITY_FACTOR = 10


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


def apply_vignette(input_img, sigma):
    """
    Given an input image and a sigma, returns a vignette of the src
    """
    height, width, _ = input_img.shape
    # Generate vignette mask using Gaussian kernels.
    kernel_x = cv2.getGaussianKernel(width, sigma)
    kernel_y = cv2.getGaussianKernel(height, sigma)

    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    # Performs the following operations sequentially: scale, calculate absolute values and convert the result
    # to 8-bit
    # expand_dims expand the shape of the mask array by adding a new dimension at the end.
    # alpha --> Contrast control (1.0 - 3.0)
    # beta  --> Brightness control (0 - 100)
    blurred = cv2.convertScaleAbs(
        input_img.copy() * np.expand_dims(mask, axis=-1), alpha=1.0, beta=0)
    return blurred


def build_blush_mask(input_img, points, color, radius):
    """
    Devise a colored mask to be added on top of the input image
    """
    mask = np.zeros_like(input_img)  # Mask that will be used for the cheeks

    # Loop throughout the specified points.
    for point in points:
        #        print('point',point)

        # Add a filled colored circle representing the blush over the mask
        mask = cv2.circle(mask, point, radius, color, cv2.FILLED)

        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        # Get the top-left coordinates of the mask area
        for i in range(0, radius):
            if (point[0] - i) > 0:
                x = point[0] - i
            if (point[1] - i) > 0:
                y = point[1] - i

#        print('x,y', x, y)

        # Vignette on the mask
        multiplier = 2
        mask[y:y + (multiplier * radius), x:x + (multiplier * radius)] = \
            apply_vignette(mask[y:y + (multiplier * radius),
                           x:x + (multiplier * radius)], sigma=20)
    return mask


def getCheekLandmarkPts(face_coordinates):
    '''
    Get cheeks landmark points
    '''
    area = [face_coordinates[i].tolist()
            #                for i in (187,50,205   #Left Cheek Area
            #                         ,425,280,411) #Right Cheek Area
            for i in (205  # Left Cheek Area
                      , 425)  # Right Cheek Area
            ]

#    print('area', area)

    right_radius = distance.euclidean(
        face_coordinates[425], face_coordinates[411])
    left_radius = distance.euclidean(
        face_coordinates[205], face_coordinates[187])
    radius = int(np.mean([left_radius, right_radius]))

    return area, radius


def process_area(input_img, area, r, g, b, radius, thickness_factor):
    blush_intensity = round(thickness_factor / BLUSH_INTENSITY_FACTOR, 1)
    print('Blush Radius', radius, 'Intensity', blush_intensity,
          'thickness_factor', thickness_factor)

    blush_mask = build_blush_mask(input_img, area, color=[
                                  r, g, b], radius=int(2*radius))
    # cv2.imshow('blush_mask', blush_mask)
    # cv2.waitKey(0)
    # BLUSH_INTENSITY
    output = cv2.addWeighted(input_img, 1.0, blush_mask, blush_intensity, 0.0)
    # cv2.imshow('output', output)
    # cv2.waitKey(0)
    return output


def apply_blush(input_path: str, display_output: bool = False):
    """
    Add blush to faces
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
        cheek_area, radius = getCheekLandmarkPts(face_coordinates)

        # Get the coordinates of the face bounding box
        x, y, w, h = faceBoundingBox
        # Extract the face -> region of interest
        # roi_face_color = frame[y:y + h, x:x + w]
        frame = process_area(frame, cheek_area, r=Rg, g=Gg, b=Bg,
                             radius=radius, thickness_factor=thickness_factor)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        # Store the blurred face in the output image
        # frame[y:y + h, x:x + w] = blushed_face
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
        label = "Applying blush"
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
    apply_blush(input_path=args['input_path'],
                display_output=args['display_output'])
