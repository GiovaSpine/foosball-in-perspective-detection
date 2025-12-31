
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

def translate_point(point: list, lower_keypoints: list):
    '''
    Docstring for translate_point
    
    :param point: Description
    :type point: list
    :param lower_keypoints: Description
    :type lower_keypoints: list
    '''

    lower_keypoints = [[1760, 478], [2916, 562], [2582, 1822], [475, 1512]]
    point = [1357, 1255]

    #lower_keypoints = [[1795, 735], [3061, 791], [2887, 1786], [251, 1511]]
    #point = [2751, 978]

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
    
    v0 = lower_keypoints[0]
    v1 = lower_keypoints[1]
    v2 = lower_keypoints[2]
    v3 = lower_keypoints[3]

    # ...
    translated_point = [0.0, 0.0]

    # step is used to update translated_point, and will be divided by 2 each time
    step = 0.5

    THRESHOLD = 0.001

    for i in range(100):

        # let's find the center of the quadrilateral
        center = calculate_intersection((v0, v2), (v1, v3))

        print(center)
        print(point)
        print(translated_point)
        print("")

        # is center close to enought to the point ?
        if np.linalg.norm(np.array(point) - np.array(center)) <= THRESHOLD:
            print("FINE")
            print(i)
            print(translated_point)
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



translate_point(0, 0)