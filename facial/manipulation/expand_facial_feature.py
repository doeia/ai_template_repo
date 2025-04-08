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

# Feature expansion factor
EXPANSION_FACTOR = 0.1

############################################################################################


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


def remove_black_border(img):
    y_nonzero, x_nonzero, _ = np.nonzero(img)
    return img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
##############################################################################################


def expand_feature(input_img, area_x, area_y, expansion_factor):
    """
    Expand face area
    """
    # Create an overlay instance
    overlay = input_img.copy()

    minX, maxX, minY, maxY = min(area_x), max(area_x), min(area_y), max(area_y)
#    print('minX,minY,maxX,maxY',minX,minY,maxX,maxY)
#    area_coordinates = np.asarray([ [minX,minY], [minX,maxY], [maxX,minY], [maxX,maxY] ])
#    print('area_coordinates', area_coordinates)

    # Extend boundaries by the expansion factor
    diffY = int((maxY - minY) * (2*expansion_factor/10))
    diffX = int((maxX - minX) * (2*expansion_factor/10))
    minY = minY - diffY
    maxY = maxY + diffY
    minX = minX - diffX
    maxX = maxX + diffX

    # Crop the region of interest
    tmp1 = overlay[minY:maxY, minX:maxX, :]
#    print('tmp1',tmp1.shape)

    # Area of interest
    Affine_Mat_W = [expansion_factor, 0, tmp1.shape[0] /
                    2.0 - expansion_factor*tmp1.shape[0]/2.0]
    Affine_Mat_H = [0, expansion_factor, tmp1.shape[1] /
                    2.0 - expansion_factor*tmp1.shape[1]/2.0]
#    print('Affine_Mat_W',Affine_Mat_W,'Affine_Mat_H',Affine_Mat_H)

    # Define a matrix for warping
    M = np.c_[Affine_Mat_W, Affine_Mat_H].T
#    print('M',M)

    tmp2 = cv2.warpAffine(tmp1, M, (tmp1.shape[1], tmp1.shape[0]))
    tmp2 = remove_black_border(tmp2)

    tmp2 = cv2.resize(tmp2, dsize=(maxX-minX, maxY-minY),
                      interpolation=cv2.INTER_AREA)
    overlay[minY:maxY, minX:maxX, :] = tmp2

    # seamlessClone
    center = (minX + ((maxX-minX)//2), minY + ((maxY-minY)//2))
    tmp2_mask = np.zeros(tmp2.shape, tmp2.dtype)
    black = np.where((tmp2_mask[:, :, 0] == 0) & (
        tmp2_mask[:, :, 1] == 0) & (tmp2_mask[:, :, 2] == 0))
    tmp2_mask[black] = (255, 255, 255)

    output = cv2.seamlessClone(tmp2, input_img, tmp2_mask, center, cv2.NORMAL_CLONE
                               # ,cv2.MIXED_CLONE
                               )
    return output


def getAreaLandmarkPts(face_coordinates, area_points):
    '''
    Get area landmark points
    '''
    area = [face_coordinates[i].tolist() for i in area_points]
    columns = list(zip(*area))
    area_x = np.array(columns[0])
    area_y = np.array(columns[1])
    return area, area_x, area_y


def expand_face_features(input_path: str, display_output: bool = False):
    """
    Expand Face Features
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
        expansion_factor = 1 + EXPANSION_FACTOR
        # Left Eye
        area_points = [130, 247, 30, 29, 27, 28, 56,
                       190, 243, 112, 26, 22, 23, 24, 110, 25]
        area, area_x, area_y = getAreaLandmarkPts(
            face_coordinates, area_points)
        frame = expand_feature(frame, area_x, area_y,
                               expansion_factor=expansion_factor)

        # Right Eye
        area_points = [463, 414, 286, 258, 257, 259, 260,
                       467, 359, 255, 339, 254, 253, 252, 256, 341]
        area, area_x, area_y = getAreaLandmarkPts(
            face_coordinates, area_points)
        frame = expand_feature(frame, area_x, area_y,
                               expansion_factor=expansion_factor)

        # Lips
        area_points = [61, 185, 40, 39, 37, 0, 267, 269, 270,
                       409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        area, area_x, area_y = getAreaLandmarkPts(
            face_coordinates, area_points)
        frame = expand_feature(frame, area_x, area_y,
                               expansion_factor=expansion_factor)
######################
    label = f"Expanding Eyes and lips by {(EXPANSION_FACTOR*100):.0f}%"

    # Output the processed image
    output_filepath = os.path.join(config.PROCESSED_PATH,
                                   str(uuid.uuid4().hex)+os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, frame)
    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER,
                   'name': os.path.basename(output_filepath), 'msg': label}
    output.append(output_item)
######################
    if display_output:
        # Display Image on screen
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
    expand_face_features(
        input_path=args['input_path'], display_output=args['display_output'])
