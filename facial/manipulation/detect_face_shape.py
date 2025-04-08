# Import Libraries
import os
import argparse
import uuid
import math
import cv2
import filetype
import config
import mediapipe
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXPANDING_FACTOR = 0.25

# To reduce false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5


def initialize_mediapipe():
    """
    Initializing mediapipe sub-modules
    """
    # Enables face detection
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)
    mpFaceModule = mediapipe.solutions.face_mesh

    return mpFaceDetection, mpFaceModule


def get_lum(image, x, y, w, h, k, gray):
    if gray == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print('image',image[-1,296])

    i1 = range(int(-w / 2), int(w / 2))
    j1 = range(0, h)

    lumar = np.zeros((len(i1), len(j1)))
    for i in i1:
        for j in j1:
            try:
                lum = np.min(image[y + k * h, x + i])
                lumar[i][j] = lum
            except:
                break
    return np.min(lumar)


def calculate_Euclidean_Distance(landmarks, pt1, pt2):
    # Calculate the Euclidean Distance between two landmarks points

    x1 = landmarks[int(pt1)][0]
    y1 = landmarks[int(pt1)][1]
    x2 = landmarks[int(pt2)][0]
    y2 = landmarks[int(pt2)][1]

    x_diff = (x1 - x2) ** 2
    y_diff = (y1 - y2) ** 2

    dist = math.sqrt(x_diff + y_diff)

    return dist


def grab_face_coordinates(img, img_landmarks):
    face_coordinates = np.array([[int(min([landmark.x * img.shape[1], img.shape[1] - 1])), int(min([landmark.y * img.shape[0], img.shape[0] - 1]))]
                                 for landmark in img_landmarks.landmark]).astype(int)
    face_coordinates[face_coordinates < 0] = 0
    return face_coordinates


def find_upper_face_point(frame, face_coordinates):
    # Find the upper face point under the hairline
    p = (face_coordinates[10][0], face_coordinates[10][1])
    x_p = p[0]
    y_p = p[1]

    gray = 0
    diff = get_lum(frame, x_p, y_p, 8, 2, -1, gray)
    limit = diff - 55
    while (diff > limit):
        y_p = int(y_p - 1)
        # print('y_p27', y_p27)
        diff = get_lum(frame, x_p, y_p, 8, 2, -1, gray)
        # cv2.circle(frame, (x_p27, y_p27), 5, color=(255, 0, 0))
    return x_p, y_p


def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)

    return x, y


def continue_line(frame, p1, p2):
    theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    end_x = int(p1[0] - 1000 * np.cos(theta))
    end_y = int(p1[1] - 1000 * np.sin(theta))
    return theta, end_x, end_y


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if b in (a, c):
        raise ValueError('Undefined angle, two identical points', (a, b, c))

    ang = ang + 360 if ang < 0 else ang
    return ang


def get_jaw_parameters(frame, face_coordinates, thickness_factor):
    """
    Get Jaw Parameters
    """
    # Calculate jaw width
    jaw_width = calculate_Euclidean_Distance(face_coordinates, 58, 288)

    # Jaw width
    cv2.line(frame, (face_coordinates[58][0], face_coordinates[58][1]), (face_coordinates[288][0], face_coordinates[288][1]), color=(255, 255, 0), thickness=thickness_factor
             )

    p1_line1 = (face_coordinates[58][0], face_coordinates[58][1])
    p2_line1 = (face_coordinates[176][0], face_coordinates[176][1])
    theta1, end_x1, end_y1 = continue_line(frame, p1_line1, p2_line1)

    p1_line2 = (face_coordinates[288][0], face_coordinates[288][1])
    p2_line2 = (face_coordinates[377][0], face_coordinates[377][1])
    theta2, end_x2, end_y2 = continue_line(frame, p1_line2, p2_line2)

    intersect = line_intersect(face_coordinates[58][0], face_coordinates[58][1], end_x1, end_y1, face_coordinates[288][0], face_coordinates[288][1], end_x2, end_y2
                               )

    cv2.circle(frame, (int(intersect[0]), int(
        intersect[1])), (thickness_factor+1), color=(255, 255, 255), thickness=-1)

    point_a = (face_coordinates[58][0], face_coordinates[58][1])
    point_b = (int(intersect[0]), int(intersect[1]))
    point_c = (face_coordinates[288][0], face_coordinates[288][1])

    cv2.line(frame, point_a, point_b, color=(0, 255, 0), thickness=thickness_factor
             )

    cv2.line(frame, point_c, point_b, color=(0, 255, 0), thickness=thickness_factor
             )

    jaw_angle = angle3pt(point_a, point_b, point_c)

    return jaw_width, jaw_angle


def calculate_face_shape_metrics(frame, face_coordinates, thickness_factor):
    ######################################
    # Calculate face shape metrics
    ######################################
    # Forehead
    forehead_width = calculate_Euclidean_Distance(face_coordinates, 54, 284)
    cv2.line(frame, (face_coordinates[54][0], face_coordinates[54][1]), (face_coordinates[284][0], face_coordinates[284][1]), color=(255, 255, 0), thickness=thickness_factor
             )

    # Cheekbones
    face_width = calculate_Euclidean_Distance(face_coordinates, 234, 454)
    cv2.line(frame, (face_coordinates[234][0], face_coordinates[234][1]), (face_coordinates[454][0], face_coordinates[454][1]), color=(255, 255, 0), thickness=thickness_factor
             )

    # Get upper face point
    x_p, y_p = find_upper_face_point(frame, face_coordinates)
    cv2.line(frame, (face_coordinates[152][0], face_coordinates[152][1]), (x_p, y_p), color=(255, 255, 0), thickness=thickness_factor
             )
    face_coordinates = np.append(face_coordinates, [[x_p, y_p]], axis=0)
    face_height = calculate_Euclidean_Distance(face_coordinates, 152, 478)

    # Chin to mouth
    chin_to_mouth_height = calculate_Euclidean_Distance(
        face_coordinates, 17, 152)
    cv2.line(frame, (face_coordinates[17][0], face_coordinates[17][1]), (face_coordinates[152][0], face_coordinates[152][1]), color=(255, 0, 0), thickness=thickness_factor
             )

    jaw_width, jaw_angle = get_jaw_parameters(
        frame, face_coordinates, thickness_factor)

    return forehead_width, face_width, face_height, chin_to_mouth_height, jaw_width, jaw_angle


def get_face_shape(forehead_width, face_width, face_height, chin_to_mouth_height, jaw_width, jaw_angle):
    print(f'Face Height={face_height:.2f}')
    print(f'Cheekbones/Face Width={face_width:.2f}')
    print(f'Forehead Width={forehead_width:.2f}')
    print(f'Jaw Width={jaw_width:.2f}')
    print(f'Chin To Mouth Height={chin_to_mouth_height:.2f}')
    print(f'Jaw Angle={jaw_angle:.2f}')

    # Face height to Face Width Ratio
    face_h_to_face_w_ratio = face_height / face_width
    print(f'Face Height to Face Width Ratio = {face_h_to_face_w_ratio:.2f}')

    shape = "Undeterministic"
    for i in range(1):
        ######################################################################################
        # Oblong or Rectangular
        if face_h_to_face_w_ratio > 1.34:
            if jaw_angle > 90:
                shape = 'Rectangular'
                # print('Rectangular. face length is largest than face width and jawline are angular ')
                break
            else:
                shape = 'Oblong'
                # print('Oblong. face length is largest than face width and jawlines are not angular')
                break
        ######################################################################################
        # Square or Round
        elif face_h_to_face_w_ratio <= 1.28:
            if jaw_angle > 90:
                shape = 'Round'
                # print('Round. face length is largest than face width and jawlines are angular')
                break
            else:
                shape = 'Square'
                # print('Square. face length is largest than face width and jawline are not angular ')
                break
        ######################################################################################
        # Pear / Heart / Oval / Diamond / Triangle
        elif face_h_to_face_w_ratio > 1.28 and face_h_to_face_w_ratio <= 1.34:
            # Face width to Jaw Width Ratio
            face_w_to_jaw_w_ratio = face_width / jaw_width
            print(
                f'Face Width to Jaw Width Ratio = {face_w_to_jaw_w_ratio:.2f}')

            # Face width to Forehead Width Ratio
            face_w_to_forehead_w_ratio = face_width / forehead_width
            print(
                f'Face Width to Forehead Width Ratio = {face_w_to_forehead_w_ratio:.2f}')

            # Jaw width to Forehead Width Ratio
            jaw_w_to_forehead_w_ratio = jaw_width / forehead_width
            print(
                f'Jaw Width to Forehead Width Ratio = {jaw_w_to_forehead_w_ratio:.2f}')

            chin_h_to_jaw_w_ratio = chin_to_mouth_height / jaw_width
            print(
                f'Chin Height to Jaw Width Ratio = {chin_h_to_jaw_w_ratio:.2f}')

            # Jaw width is larger or close to face width
            if face_w_to_jaw_w_ratio < 1.1:
                shape = 'Pear'
                # print('Pear. Jaw width is wider or very close to face width.')
                break

            if jaw_angle > 90 and chin_h_to_jaw_w_ratio > 0.2 and face_w_to_forehead_w_ratio > 1.11:
                shape = 'Oval'
                # print('Oval.')
                break

            if jaw_angle < 90 and chin_h_to_jaw_w_ratio <= 0.2 and face_w_to_forehead_w_ratio < 1.11:
                shape = 'Heart'
                # print('Heart.')
                break

            # if face_w_to_forehead_w_ratio < 1.11 and face_w_to_jaw_w_ratio < 1.2 and jaw_w_to_forehead_w_ratio < 0.9:
            if forehead_width < face_width and face_width > jaw_width:
                shape = 'Diamond'
                # print('Diamond.')
                break

            # if face_w_to_forehead_w_ratio < 1.11 and face_w_to_jaw_w_ratio < 1.2 and jaw_w_to_forehead_w_ratio > 0.9:
            if forehead_width > face_width < jaw_width \
                    or forehead_width < face_width < jaw_width:
                shape = 'Triangular'
                # print('Triangular.')
                break

    print('shape = ', shape)

    params = {
        "Face Height": face_height, "Face Width": face_width, "Forehead Width": forehead_width, "Jaw Width": jaw_width, "Chin To Mouth Height": chin_to_mouth_height, "Jaw Angle": jaw_angle
        # ,"Face Width to Jaw Width Ratio": face_w_to_jaw_w_ratio
        # ,"Face Width to Forehead Width Ratio": face_w_to_forehead_w_ratio
        # ,"Jaw width to Forehead Width Ratio": jaw_w_to_forehead_w_ratio
        # ,"Chin Height to Jaw Width Ratio":chin_h_to_jaw_w_ratio
    }

    return params, shape


def layout_shape_parameters(params):
    """
    Layout list of parameters in an image
    """
    parameters_img = Image.new('RGB', (300, 300), color=(0, 0, 0, 0))
    # Get a drawing context
    draw = ImageDraw.Draw(parameters_img)
    font = ImageFont.truetype(os.path.join(
        config.FONTS_PATH, 'OpenSansEmoji.ttf'), 20, encoding='unic')
    # print(output)
    for idx, (i) in enumerate(params):
        label = "{}={:.2f}".format(i, params.get(i))
        draw.text((20, 40 * idx), label, font=font, embedded_color=True)
    parameters_img = np.array(parameters_img)
    return parameters_img


def enlarge_bounding_box(x, y, w, h):
    """
    Enlarge the bounding box based on the expanding factor
    """
    # create a larger bounding box with buffer around keypoints
    x1 = int(x - EXPANDING_FACTOR * w)
    w1 = int(w + 2 * EXPANDING_FACTOR * w)
    y1 = int(y - EXPANDING_FACTOR * h)
    h1 = int(h + 2 * EXPANDING_FACTOR * h)
    # print('x1,y1,w1,h1', x1, y1, w1, h1)
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    # print('x1,y1,w1,h1', x1, y1, w1, h1)
    return x1, y1, w1, h1


def calculate_thickness_factor(frame):
    thickness_factor = (frame.shape[0] * frame.shape[1]) / \
        ((frame.shape[0] + frame.shape[1]) * 100)
    thickness_factor = 1 if thickness_factor < 1 else int(thickness_factor)
    return thickness_factor


def detect_face_shape(input_path: str, display_output: bool = False):
    """
    Detecting the shapes of the faces showing within the image
    """
    # Initialize the mediapipe module
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

    # info, success, warning, danger
    output_msg = {'msg': "{} face(s) detected.".format(
        len(faces.detections)), 'category': "info"}
    output_info.append(output_msg)

    mpFaceMesh = mpFaceModule.FaceMesh(
        static_image_mode=True, max_num_faces=len(faces.detections), refine_landmarks=True, min_detection_confidence=MIN_CONFIDENCE_LEVEL
    )
    landmarks = mpFaceMesh.process(rgb_frame).multi_face_landmarks
    # Calculate the adequate thickness factor based on the image resolution.
    thickness_factor = calculate_thickness_factor(frame)

    # Loop over the faces detected
    for idx, (face_detected, face_landmarks) in enumerate(zip(faces.detections, landmarks)):

        output_item = None
        label = f"Face ID = {(idx+1)} - Detection Score {int(face_detected.score[0]*100)}%"
        print(label)

        face_coordinates = grab_face_coordinates(frame, face_landmarks)

        relativeBoundingBox = face_detected.location_data.relative_bounding_box
        frameHeight, frameWidth, frameChannels = frame.shape
        faceBoundingBox = int(relativeBoundingBox.xmin*frameWidth), int(relativeBoundingBox.ymin*frameHeight), int(
            relativeBoundingBox.width*frameWidth), int(relativeBoundingBox.height*frameHeight)

        x, y, w, h = faceBoundingBox
        face_center_point = ((x+w)//2, (y+h)//2)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Create a larger bounding box with buffer around keypoints
        x1, y1, w1, h1 = enlarge_bounding_box(x, y, w, h)
        # cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 255), 3)

        # Getting Face Shape Parameters
        forehead_width, face_width, face_height, chin_to_mouth_height, jaw_width, jaw_angle = calculate_face_shape_metrics(
            frame, face_coordinates, thickness_factor)

        params, shape = get_face_shape(
            forehead_width, face_width, face_height, chin_to_mouth_height, jaw_width, jaw_angle)
        if shape:
            shape_parameters_layout = layout_shape_parameters(params)
            combined_img = cv2.resize(frame, (400, 400), cv2.INTER_CUBIC)
            roi_face_color = frame[y1:y1 + h1, x1:x1 + w1]
            if shape_parameters_layout is not None:
                combined_img = cv2.hconcat([cv2.resize(roi_face_color, (400, 400), cv2.INTER_CUBIC), cv2.resize(
                    shape_parameters_layout, (400, 400), cv2.INTER_CUBIC)])

                label = label + '-- Shape = ' + shape
                output_filepath = os.path.join(config.PROCESSED_PATH, str(
                    uuid.uuid4().hex)+os.path.splitext(input_path)[1])
                cv2.imwrite(output_filepath, combined_img)
                output_item = {'id': idx, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(output_filepath), 'msg': label
                               }
                output.append(output_item)

        if display_output:
            # Display Image on screen
            cv2.imshow(shape, combined_img)
            # Mantain output until user presses a key
            cv2.waitKey(0)

    if display_output:
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
    detect_face_shape(
        input_path=args['input_path'], display_output=args['display_output'])
