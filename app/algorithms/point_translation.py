
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



def find_t(A, B, point, eps=1e-9):
    '''
    Docstring for find_t
    
    :param A: Description
    :param B: Description
    :param P: Description
    :param eps: Description
    '''
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    P = np.array(point, dtype=float)

    AB = A - B
    PB = P - B

    # check if A and B are not to close
    non_zero = np.abs(AB) > eps
    if not np.any(non_zero):
        raise ValueError("A and B are too close")

    t_components = PB[non_zero] / AB[non_zero]

    t = np.mean(t_components)
    return t



def translate_point(point: list, lower_keypoints: list):
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

    # let's find the center of the quadrilateral
    center = calculate_intersection((lower_keypoints[0], lower_keypoints[2]), (lower_keypoints[1], lower_keypoints[3]))

    # let's find the intersection between the line from the point to the vanishing_point_y
    # and the line from the center to the vanishing_point_x
    x_projection = calculate_intersection((point, vanishing_point_y), (center, vanishing_point_x))

    # let's find the intersection between the line from the point to the vanishing_point_x
    # and the line from the center to the vanishing_point_y
    y_projection = calculate_intersection((point, vanishing_point_x), (center, vanishing_point_y))

    # the center of each side
    side_0_3_center = calculate_intersection((lower_keypoints[0], lower_keypoints[3]), (center, vanishing_point_x))
    side_1_2_center = calculate_intersection((lower_keypoints[1], lower_keypoints[2]), (center, vanishing_point_x))
    side_0_1_center = calculate_intersection((lower_keypoints[0], lower_keypoints[1]), (center, vanishing_point_y))
    side_2_3_center = calculate_intersection((lower_keypoints[2], lower_keypoints[3]), (center, vanishing_point_y))

    # let's look if center to x_projection has the same direction of center to side_0_3_center
    if np.dot((np.array(x_projection) - np.array(center)), (np.array(side_0_3_center) - np.array(center))) > 0.0:
        # we know that x_projection is a point in the line from center to side_0_3_center
        # and that the translated x position is negative
        side_x = side_0_3_center
        sign_x = False
    else:
        # we know that x_projection is a point in the line from center to side_1_2_center
        # and that the translated x position is positive
        side_x = side_1_2_center
        sign_x = True
    
    # let's look if center to y_projection has the same direction of center to side_0_1_center
    if np.dot((np.array(y_projection) - np.array(center)), (np.array(side_0_1_center) - np.array(center))) > 0.0:
        # we know that y_projection is a point in the line from center to side_0_1_center
        # and that the translated y position is positive
        side_y = side_0_1_center
        sign_y = True
    else:
        # we know that y_projection is a point in the line from center to side_2_3_center
        # and that the translated y position is negative
        side_y = side_2_3_center
        sign_y = False

    # let's use the linear interpolation: t*A + (1 - t)*B for t in [0, 1]

    # Tx * side_x + (1 - Tx) * center = projection_x
    # Tx is the normalized x position in the reference coordinate system
    tx = find_t(B=center, A=side_x, point=x_projection)
    if not sign_x: tx *= -1.0  # sign adjustment

    # Ty * side_y + (1 - Ty) * center = projection_y
    # Ty is the normalized y position in the reference coordinate system
    ty = find_t(B=center, A=side_y, point=y_projection)
    if not sign_y: ty *= -1.0  # sign adjustment

    return [tx, ty], None

