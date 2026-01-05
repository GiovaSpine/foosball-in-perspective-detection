
import numpy as np

def is_convex_quadrilateral(points):
    '''
    ASSUMING they are ordered!!!
    '''
    pts = np.asarray(points)

    if pts.shape != (4, 2):
        raise ValueError("Error: 4 bi-dimensional points are needed")

    # a polygon is convex if all vectorial consecutive products have the same sign
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = []

    for i in range(4):
        o = pts[i]
        a = pts[(i + 1) % 4]
        b = pts[(i + 2) % 4]

        z = cross(o, a, b)
        if z != 0:
            signs.append(np.sign(z))

    return all(s == signs[0] for s in signs)


def is_point_in_quadrilateral(point, quad):
    '''
    ...
    '''
    p = np.array(point)
    q = np.array(quad)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = []

    for i in range(4):
        a = q[i]
        b = q[(i + 1) % 4]

        val = cross(a, b, p)

        if val != 0:
            signs.append(np.sign(val))

    if not signs:
        return False

    return all(s == signs[0] for s in signs)


def calculate_intersection(line1: tuple, line2: tuple) -> tuple:
    '''
    Calculate the intersection of 2 lines, each rapresented as 2 points in 2d.

    Parameters:
    line1 (tuple): The first line as (x1, y1), (x2, y2)
    line2 (tuple): The second line as (x1, y1), (x2, y2)

    Returns:
    tuple: The point of intersection if it exists
    '''
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def determinant(a, b):
        return a[0] * b[1] - a[1] * b[0]

    divisor = determinant(x_diff, y_diff)
    if divisor == 0:
        raise ValueError(f"The lines do not intersect. {line1}, {line2}")

    d = (determinant(*line1), determinant(*line2))
    x = determinant(d, x_diff) / divisor
    y = determinant(d, y_diff) / divisor
    return x, y

# ---------------------------------------------------------

def translate_point(point: list, lower_keypoints: list, max_iterations: int=100) -> tuple:
    '''
    Docstring for translate_point
    
    :param point: Description
    :type point: list
    :param lower_keypoints: Description
    :type lower_keypoints: list
    '''

    # let's check if the lower_keypoints generate a valid area: they should generate a 4 sided convex polygon
    if not is_convex_quadrilateral(lower_keypoints):
        return None, "The quadrilateral formed by the lower keypoints is not convex"

    # let's check if the point is inside the quadrilateral 
    if not is_point_in_quadrilateral(point, lower_keypoints):
        return None, "The point to translate is not inside the quadrilateral formed by the lower keypoints"
    
    # at this point we can translate

    # we need the vanishing points
    vanishing_point_x = calculate_intersection((lower_keypoints[0], lower_keypoints[1]), (lower_keypoints[2], lower_keypoints[3]))
    vanishing_point_y = calculate_intersection((lower_keypoints[0], lower_keypoints[3]), (lower_keypoints[1], lower_keypoints[2]))
    
    # vertices of the chunk of the play area, following the same order of the lower_keypoints
    v0 = lower_keypoints[0]
    v1 = lower_keypoints[1]
    v2 = lower_keypoints[2]
    v3 = lower_keypoints[3]

    # translated point starting in (0, 0)
    translated_point = [0.0, 0.0]

    # step is used to update translated_point, and will be divided by 2 each time
    step = 0.5

    # the min threasold between the center and point to quit the algorithm
    THRESHOLD = 0.001

    for _ in range(max_iterations):

        # let's find the center of the quadrilateral
        center = calculate_intersection((v0, v2), (v1, v3))

        # is center close to enough to the point ?
        if np.linalg.norm(np.array(point) - np.array(center)) <= THRESHOLD:
            break

        # let's find the intersection between the line from the point to the vanishing_point_y
        # and the line from the center to the vanishing_point_x
        x_projection = calculate_intersection((point, vanishing_point_y), (center, vanishing_point_x))

        # let's find the intersection between the line from the point to the vanishing_point_x
        # and the line from the center to the vanishing_point_y
        y_projection = calculate_intersection((point, vanishing_point_x), (center, vanishing_point_y))

        # the center of each side
        side_0_3_center = calculate_intersection((v0, v3), (center, vanishing_point_x))
        side_1_2_center = calculate_intersection((v1, v2), (center, vanishing_point_x))
        side_0_1_center = calculate_intersection((v0, v1), (center, vanishing_point_y))
        side_2_3_center = calculate_intersection((v2, v3), (center, vanishing_point_y))

        # let's look if center to x_projection has the same direction of center to side_0_3_center
        if np.dot((np.array(x_projection) - np.array(center)), (np.array(side_0_3_center) - np.array(center))) > 0.0:
            # x_projection relative to center is negative
            sign_x = False
        else:
            # x_projection relative to center is positive
            sign_x = True
        
        # let's look if center to y_projection has the same direction of center to side_0_1_center
        if np.dot((np.array(y_projection) - np.array(center)), (np.array(side_0_1_center) - np.array(center))) > 0.0:
            # y_projection relative to center is positive
            sign_y = True
        else:
            # y_projection relative to center is negative
            sign_y = False

        # at this point we know the quadrant where the point is located
        # let's grab the vertices of the quadrant that contains the point

        if (sign_x, sign_y) == (True, True):
            # top right quadrant
            v0 = side_0_1_center
            v1 = v1
            v2 = side_1_2_center
            v3 = center
        elif (sign_x, sign_y) == (False, True):
            # top left quadrant
            v0 = v0
            v1 = side_0_1_center
            v2 = center
            v3 = side_0_3_center
        elif (sign_x, sign_y) == (True, False):
            # bottom right quadrant
            v0 = center
            v1 = side_1_2_center
            v2 = v2
            v3 = side_2_3_center
        else:
            # bottom left quadrant
            v0 = side_0_3_center
            v1 = center
            v2 = side_2_3_center
            v3 = v3

        # let's update the translated_point
        if sign_x:
            translated_point[0] += step
        else:
            translated_point[0] -= step
        if sign_y:
            translated_point[1] += step
        else:
            translated_point[1] -= step

        # update step
        step /= 2.0

    return translated_point, None


def calculate_player_lines(keypoints: list) -> tuple:
    '''
    Docstring for calculate_player_lines
    
    :param keypoints: Description
    :type keypoints: list
    :return: Description
    :rtype: list
    '''

    if not is_convex_quadrilateral([keypoints[0], keypoints[4], keypoints[7], keypoints[3]]):
        return None, "The quadrilateral of the left side is not convex"
    
    if not is_convex_quadrilateral([ keypoints[5], keypoints[1], keypoints[2], keypoints[6]]):
        return None, "The quadrilateral of the right side is not convex"

    # we have to divide the face given by the keypoints 0, 3, 4, 7 (and 1, 2, 5, 6) into 8 parts
    # we will do it by finding centers, to follow the perspective

    N_DIVISIONS = 3

    vp_z_1 = calculate_intersection((keypoints[0], keypoints[4]), (keypoints[1], keypoints[5]))
    vp_z_2 = calculate_intersection((keypoints[3], keypoints[7]), (keypoints[2], keypoints[6]))
    # vanishing_point_z_1 should in theory be equal to vanishing_point_z_2; in practice we take the average
    vanishing_point_z = ((vp_z_1[0] + vp_z_2[0]) / 2.0, (vp_z_1[1] + vp_z_2[1]) / 2.0)

    def get_center(centers, iteration, v0, v1, v2, v3):
        '''
        Docstring for get_center
        
        :param v0: Description
        :param v1: Description
        :param v2: Description
        :param v3: Description
        '''
        center = calculate_intersection((v0, v2), (v1, v3))
        
        upper_side_center = calculate_intersection((center, vanishing_point_z), (v0, v3))
        lower_side_center = calculate_intersection((center, vanishing_point_z), (v1, v2))

        if iteration < N_DIVISIONS:
            get_center(centers, iteration + 1, v0, v1, lower_side_center, upper_side_center)
            get_center(centers, iteration + 1, upper_side_center, lower_side_center, v2, v3)
        else:
            # we reached the desired level
            centers.append(center)

    centers_face_1 = []
    centers_face_2 = []
    # left face given by 0, 3, 4, 7 keypoints
    get_center(centers_face_1, 0, keypoints[0], keypoints[4], keypoints[7], keypoints[3])
    # right face given by 1, 2, 5, 6
    get_center(centers_face_2, 0, keypoints[5], keypoints[1], keypoints[2], keypoints[6])

    # player lines is a list formed by [p1, p2] where p1 is the point of the left face, and p2 of the right face
    player_lines = [[p1, p2] for p1, p2 in zip(centers_face_1, centers_face_2)]

    return player_lines, None


def keypoints_cleaning(keypoints):

    # we will assume that the vanishing point for the z axis given by the lines 0_4 and 1_5 is correct
    # then we will check if the lines 2_6 and 3_7 tends to go towards that vanshing point

    vp_z_1 = calculate_intersection((keypoints[0], keypoints[4]), (keypoints[1], keypoints[5]))
    vp_z_2 = calculate_intersection((keypoints[2], keypoints[5]), (keypoints[6], keypoints[7]))

    MAX_VPS_Z_DISTANCE = 2500  # found empirically
    
    if np.linalg.norm(np.array(vp_z_1) - vp_z_2) > MAX_VPS_Z_DISTANCE:
        # considering vp_z_1 correct
        # we have to recompute the keypoint 6 and 7

        vp_y = calculate_intersection((keypoints[0], keypoints[3]), (keypoints[1], keypoints[2]))
        vp_x = calculate_intersection((keypoints[0], keypoints[1]), (keypoints[2], keypoints[3]))

        MIN_DEGREES_DIFFERENCE = 5.0

        # for the keypoint 6 the idea is to:
        # find the intersection between the line that goes from 5 to vp_y and the line that goes from 2 to vp_z_1

        vector_6_to_z = np.array(vp_z_1) - np.array(keypoints[6])
        vector_6_to_z_degrees = np.degrees(np.arctan2(vector_6_to_z[1], vector_6_to_z[0]))

        vector_5_to_y = np.array(keypoints[5]) - np.array(vp_y)
        vector_5_to_y_degrees = np.degrees(np.arctan2(vector_5_to_y[1], vector_5_to_y[0]))

        if abs(vector_6_to_z_degrees - vector_5_to_y_degrees) < MIN_DEGREES_DIFFERENCE:
            # the angles are to similar (the two lines are similar)
            # it happens when the face on the right side is barely visible
            # we can find the intersection between the line that goes from 7 to vp_x and the line that goes from 2 to vp_z_1
            new_keypoint_6 = calculate_intersection((keypoints[7], vp_x), (keypoints[2], vp_z_1))
        else:
            # we can compute with the intersection between the line that goes from 5 to vp_y and the line that goes from 2 to vp_z_1
            new_keypoint_6 = calculate_intersection((keypoints[5], vp_y), (keypoints[2], vp_z_1))

        # for the keypoint 7 the idea is to:
        # find the intersection between the line that goes from 4 to vp_y and the line that goes from 3 to vp_z_1

        vector_7_to_z = np.array(vp_z_1) - np.array(keypoints[7])
        vector_7_to_z_degrees = np.degrees(np.arctan2(vector_7_to_z[1], vector_7_to_z[0]))

        vector_4_to_y = np.array(keypoints[4]) - np.array(vp_y)
        vector_4_to_y_degrees = np.degrees(np.arctan2(vector_4_to_y[1], vector_4_to_y[0]))

        if abs(vector_7_to_z_degrees - vector_4_to_y_degrees) < MIN_DEGREES_DIFFERENCE:
            # the angles are to similar (the two lines are similar)
            # it happens when the face on the right side is barely visible
            # we can find the intersection between the line that goes from 6 to vp_x and the line that goes from 3 to vp_z_1
            new_keypoint_7 = calculate_intersection((keypoints[6], vp_x), (keypoints[3], vp_z_1))
        else:
            # we can compute with the intersection between the line that goes from 4 to vp_y and the line that goes from 3 to vp_z_1
            new_keypoint_7 = calculate_intersection((keypoints[4], vp_y), (keypoints[3], vp_z_1))
 
        keypoints[6] = new_keypoint_6
        keypoints[7] = new_keypoint_7

        print(keypoints[6], keypoints[7])

    return keypoints, None
