
import numpy as np

# TYPE CHECKING

def check_keypoints(keypoints: list):
    '''
    Checks if keypoint is a list of 8 elements that are lists of 2 floats.

    Parameter:
    keypoints (list): The list of keypoints to check

    Returns:
    bool: Wheter the keypoints respect the described type
    '''
    # keypoints has to be a list
    if not isinstance(keypoints, list):
        raise TypeError(f"keypoints needs to be a list: {type(keypoints)} given")
    # keypoints has to have 8 elements
    if len(keypoints) != 8:
        raise ValueError(f"keypoints needs to have 8 elements: {len(keypoints)} given")
    # every element has to be a list or tuple of 2 floats or ints
    for kp in keypoints:
        if not isinstance(kp, (list, tuple)):
            raise TypeError(f"keypoints needs to contain lists: {type(kp)} given")
        if len(kp) != 2:
            raise ValueError(f"the lists or tuples of keypoints need to have 2 elements: {len(kp)} given")
        if not all(isinstance(x, (float, int)) for x in kp):
            raise ValueError(f"the lists or tuples of keypoints need to have 2 floats or ints")


def check_quadrilateral(quadrilateral: list) -> None:
    '''
    Checks if quadrilateral is a list of 4 elements that are lists of 2 floats.

    Parameter:
    quadrilateral (list): The quadrilateral to check

    Returns:
    None
    '''
    # quadrilateral has to be a list
    if not isinstance(quadrilateral, list):
        raise TypeError(f"quadrilateral needs to be a list: {type(quadrilateral)} given")
    # quadrilateral has to have 4 elements
    if len(quadrilateral) != 4:
        raise ValueError(f"quadrilateral needs to have 4 elements: {len(quadrilateral)} given")
    # every element has to be a list or tuple of 2 floats or ints
    for p in quadrilateral:
        if not isinstance(p, (list, tuple)):
            raise TypeError(f"quadrilateral needs to contain lists: {type(p)} given")
        if len(p) != 2:
            raise ValueError(f"the lists or tuples of quadrilateral need to have 2 elements: {len(p)} given")
        if not all(isinstance(x, (float, int, np.floating, np.integer)) for x in p):
            raise ValueError(f"the lists or tuples of quadrilateral need to have 2 floats or ints")


def check_point(point: list | tuple | np.ndarray) -> None:
    '''
    Checks if point is a list of 2 floats.

    Parameter:
    point (list or tuple): The point to check

    Returns:
    None
    '''
    if not isinstance(point, (list, tuple, np.ndarray)):
        raise TypeError(f"point needs to be a list or tuple: {type(point)} given")
    if len(point) != 2:
        raise ValueError(f"point needs to contain 2 elements: {len(point)} given")
    if not all(isinstance(x, (float, int, np.floating, np.integer)) for x in point):
        raise TypeError(f"point needs to contain 2 floats or ints")


def check_line(line: list | tuple) -> None:
    '''
    Checks if line is a list of 2 lists that each contain 2 floats.

    Parameter:
    line (list or tuple): The line in the form [p1, p2] or (p1, p2) where p1 and p2 are bidimensional points

    Returns:
    None
    '''
    if not isinstance(line, (list, tuple)):
        raise TypeError("line needs to be a list or tuple")
    if len(line) != 2:
        raise ValueError("line needs to contain 2 elements")
    check_point(line[0])
    check_point(line[1])

# ---------------------------------------------------------

# ERRORS

class NonConvexQuadrilateralError(ValueError):
    pass

class PointOutsideQuadrilateralError(ValueError):
    pass

class AlgorithmFailedError(ValueError):
    pass

# ---------------------------------------------------------

# UTILITY FUNCTIONS

def cross(o: list | tuple, a: list | tuple, b: list | tuple) -> float:
    '''
    Calculates the 2D vector product of the vectors OA and OB
    
    Parameters:
    o (list or tuple): Origin point O as [x, y] or (x, y)
    a (list or tuple): Point A as [x, y] or (x, y)
    b (list or tuple):  Point B as [x, y] or (x, y)

    Returns:
    float: The signed scalar representing the 2D cross product of OA and OB
    '''
    # check parameters
    check_point(o)
    check_point(a)
    check_point(b)

    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex_quadrilateral(quadrilateral: list) -> bool:
    '''
    Checks if the quadrilateral is convex.

    Parameters:
    quadrilateral (list): The list of 4 points (lists of 2 floats) that forms the quadrilateral

    Returns:
    bool: Wheter the quadrilateral is convex or not
    '''
    # check parameters
    check_quadrilateral(quadrilateral)

    pts = np.asarray(quadrilateral)

    # a polygon is convex if all vectorial consecutive products have the same sign

    signs = []

    for i in range(4):
        o = pts[i]
        a = pts[(i + 1) % 4]
        b = pts[(i + 2) % 4]

        z = cross(o, a, b)
        if z != 0:
            signs.append(np.sign(z))

    return all(s == signs[0] for s in signs)


def is_point_in_quadrilateral(point: list | tuple, quadrilateral: list) -> bool:
    '''
    Checks if the point is inside the quadliateral.

    Parameters:
    point (list or tuple): The bidimensional point to check
    quadrilateral (list): The quadrilater that should contain the point or not

    Returns:
    bool: Wheter the point is inside the quadrilateral or not
    '''

    # check parameters
    check_point(point)
    check_quadrilateral(quadrilateral)

    p = np.array(point)
    q = np.array(quadrilateral)

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

    # check parameters
    check_line(line1)
    check_line(line2)

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

# ALGORITHMS

def translate_point(point: list | tuple | np.ndarray, lower_keypoints: list, threasold: float=0.001, max_iterations: int=100) -> list:
    '''
    Converts a point from the coordinate system of the image to the coordinate system of the quadrilateral.
    Given lower_keypoints: the 4 point that form the quadrilateral in the image coordinate system,
    abd point: the point to translate that should be inside the quadrilateral, in the image coordinate system,
    the function translate the coordinates of the point in the image to the ones in the quadrilateral.

    - The coordinate system in the quadrilater is in the center (found by intersecting the line from the keypoint 0->2 and 1->3)
    - X goes from the center towards the line from the keypoint 1->2
    - Y goes from the center towards the line from the keypoint 0->1

    The translated point produced is normalized in respect of the 2 lines (keypoint 1->2 and keypoint 0->1)
    
    Parameters:
    point (list, tuple or np.ndarray): The point, in the image coordinate system, to translate
    lower_keypoints (list): The keypoints that form the quadrilateral
    threasold (float): The min distance the calculated translated point should have from the real one
    max_iterations (int): The max amount of iterations the algorithm should do

    Returns:
    list: The translated point as [x, y]
    '''
    # check parameters
    check_point(point)
    check_quadrilateral(lower_keypoints)
    if not isinstance(threasold, float):
        raise TypeError("threasold has to be a float")
    if threasold <= 0.0:
        raise ValueError("threasold has to be > 0")
    if not isinstance(max_iterations, int):
        raise TypeError("max_iterations has to be a int")
    if max_iterations < 1:
        raise ValueError("max_iterations has to be > 0")

    # let's check if the lower_keypoints generate a valid area: they should generate a 4 sided convex polygon
    if not is_convex_quadrilateral(lower_keypoints):
        raise NonConvexQuadrilateralError(
            "The quadrilateral formed by the lower keypoints is not convex"
        )

    # let's check if the point is inside the quadrilateral 
    if not is_point_in_quadrilateral(point, lower_keypoints):
        raise PointOutsideQuadrilateralError(
            "The point to translate is not inside the quadrilateral formed by the lower keypoints"
        )
    
    # at this point we can translate...

    # we need the vanishing points
    try:
        vanishing_point_x = calculate_intersection((lower_keypoints[0], lower_keypoints[1]), (lower_keypoints[2], lower_keypoints[3]))
        vanishing_point_y = calculate_intersection((lower_keypoints[0], lower_keypoints[3]), (lower_keypoints[1], lower_keypoints[2]))
    except Exception as e:
        raise AlgorithmFailedError("Algorithm failed (unable to calculate vanishing points)") from e
    
    # vertices of the chunk of the play area, following the same order of the lower_keypoints
    v0 = lower_keypoints[0]
    v1 = lower_keypoints[1]
    v2 = lower_keypoints[2]
    v3 = lower_keypoints[3]

    # translated point starting in (0, 0)
    translated_point = [0.0, 0.0]

    # step is used to update translated_point, and will be divided by 2 each time
    step = 0.5

    for _ in range(max_iterations):

        # let's find the center of the quadrilateral
        center = calculate_intersection((v0, v2), (v1, v3))

        # is center close to enough to the point ?
        if np.linalg.norm(np.array(point) - np.array(center)) <= threasold:
            break
        
        try:
            # let's find the intersection between the line from the point to the vanishing_point_y
            # and the line from the center to the vanishing_point_x
            x_projection = calculate_intersection((point, vanishing_point_y), (center, vanishing_point_x))

            # let's find the intersection between the line from the point to the vanishing_point_x
            # and the line from the center to the vanishing_point_y
            y_projection = calculate_intersection((point, vanishing_point_x), (center, vanishing_point_y))
        except Exception as e:
            raise AlgorithmFailedError("Algorithm failed (unable to calculate x, y projections)") from e

        try:
            # the center of each side
            side_0_3_center = calculate_intersection((v0, v3), (center, vanishing_point_x))
            side_1_2_center = calculate_intersection((v1, v2), (center, vanishing_point_x))
            side_0_1_center = calculate_intersection((v0, v1), (center, vanishing_point_y))
            side_2_3_center = calculate_intersection((v2, v3), (center, vanishing_point_y))
        except Exception as e:
            raise AlgorithmFailedError("Algorithm failed (unable to calculate centers of chunk's sides)") from e

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

    return translated_point


def calculate_player_lines(keypoints: list) -> list:
    '''
    Calculates the player lines (the 8 lines where the are the small statues of football players)

    :param keypoints: Description
    :type keypoints: list
    :return: Description
    :rtype: list
    '''

    # check parameters
    check_keypoints(keypoints)

    if not is_convex_quadrilateral([keypoints[0], keypoints[4], keypoints[7], keypoints[3]]):
        raise NonConvexQuadrilateralError("The quadrilateral of the left side is not convex")
    
    if not is_convex_quadrilateral([ keypoints[5], keypoints[1], keypoints[2], keypoints[6]]):
        raise NonConvexQuadrilateralError("The quadrilateral of the right side is not convex")

    # we have to divide the face given by the keypoints 0, 3, 4, 7 (and 1, 2, 5, 6) into 8 parts
    # we will do it by finding centers, to follow the perspective

    N_DIVISIONS = 3

    try:
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

    except Exception:
        raise AlgorithmFailedError("Unable to calculate player lines")

    return player_lines


def keypoints_cleaning(keypoints: list) -> list:
    '''
    '''
    # check parameteres
    check_keypoints(keypoints)

    # we will assume that the vanishing point for the z axis given by the lines 0_4 and 1_5 is correct
    # then we will check if the lines 2_6 and 3_7 tends to go towards that vanshing point
    try:
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
    except Exception:
        raise AlgorithmFailedError("Unable to clean keypoints")

    return keypoints

