import numpy as np

def find_min_max_values(x, y, z):
    x_min_index = list(x).index(min(x))
    x_max_index = list(x).index(max(x))
    y_min_index = list(y).index(min(y))
    y_max_index = list(y).index(max(y))
    z_min_index = list(z).index(min(z))
    z_max_index = list(z).index(max(z))

    xmin_point = (x[x_min_index], y[x_min_index], z[x_min_index])
    xmax_point = (x[x_max_index], y[x_max_index], z[x_max_index])
    ymin_point = (x[y_min_index], y[y_min_index], z[y_min_index])
    ymax_point = (x[y_max_index], y[y_max_index], z[y_max_index])
    zmin_point = (x[z_min_index], y[z_min_index], z[z_min_index])
    zmax_point = (x[z_max_index], y[z_max_index], z[z_max_index])

    return xmin_point, xmax_point, ymin_point, ymax_point, zmin_point, zmax_point

def distance(p1, p2): 
    return np.sqrt(np.sum([(j-i)**2 for i, j in zip(p1, p2)]))
    
def extract_variables_from_las_object(las):
    x = np.array(list(las.X))
    y = np.array(list(las.Y))
    z = np.array(list(las.Z))
    rgb = np.array([(r/255, g/255, b/255) for r, g, b in zip(las.red, las.green, las.blue)])

    # Used to adjust the size of the scatterplot points
    ys = np.random.randint(130, 195, len(x))
    zs = np.random.randint(30, 160, len(y))
    s = zs / ((ys * 0.01) ** 2)

    return x, y, z, s, rgb, las.point_source_id

def is_equal_color(rgb1, rgb2):
    return rgb1[0]==rgb2[0] and rgb1[1]==rgb2[1] and rgb1[2]==rgb2[2]

def intersect_2planes(plane1, plane2):
    A1, B1, C1, D1 = plane1
    A2, B2, C2, D2 = plane2
    
    A = np.array([[A1, B1, C1], [A2, B2, C2]])
    b = np.array([-D1, -D2])

    intersection = np.dot(np.linalg.inv(A[:,:2]), b)
    double_intersection = np.append(intersection, 0, axis=None)

    direction_vector = np.cross(A[0], A[1])

    return double_intersection, direction_vector

def intersect_3planes(plane1, plane2, plane3, z_value=False):
    A1, B1, C1, D1 = plane1
    A2, B2, C2, D2 = plane2
    A3, B3, C3, D3 = plane3

    A = np.array([[A1, B1, C1], [A2, B2, C2], [A3, B3, C3]])
    b = np.array([-D1, -D2, -D3])

    triple_intersection = np.dot(np.linalg.pinv(A), b)

    # Find intersection points where z=0
    if z_value:
        A1, A2, A3 = A[:2,:2], A[1:,:2], np.array([A[0][:2], A[2][:2]])
        b1, b2, b3 = b[:2], b[1:], np.array([b[0], b[2]])

        ip1 = np.dot(np.linalg.pinv(A1), b1)
        ip2 = np.dot(np.linalg.pinv(A2), b2)
        ip3 = np.dot(np.linalg.pinv(A3), b3)

        dv1 = np.cross(A[0], A[1])
        dv2 = np.cross(A[1], A[2])
        dv3 = np.cross(A[0], A[2])

        double_intersections = [np.append(ip1, 0, axis=None), np.append(ip2, 0, axis=None), np.append(ip3, 0, axis=None)]
        
        direction_vectors = [dv1, dv2, dv3]

        double_intersections = [p + z_value/dv[2] * dv for p, dv in zip(double_intersections, direction_vectors) if abs(dv[2]) > 0.1]

        return triple_intersection, double_intersections, direction_vectors
    else:
        A1, A2, A3 = A[:2,:2], A[1:,:2], np.array([A[0][:2], A[2][:2]])
        b1, b2, b3 = b[:2], b[1:], np.array([b[0], b[2]])
        ip1 = np.dot(np.linalg.pinv(A1), b1)
        ip2 = np.dot(np.linalg.pinv(A2), b2)
        ip3 = np.dot(np.linalg.pinv(A3), b3)

        dv1 = np.cross(A[0], A[1])
        dv2 = np.cross(A[1], A[2])
        dv3 = np.cross(A[0], A[2])

        double_intersections = [np.append(ip1, 0, axis=None), np.append(ip2, 0, axis=None), np.append(ip3, 0, axis=None)]
        direction_vectors = [dv1, dv2, dv3]

        return triple_intersection, double_intersections, direction_vectors
    
def intersect_4planes(plane1, plane2, plane3, plane4):
    A1, B1, C1, D1 = plane1
    A2, B2, C2, D2 = plane2
    A3, B3, C3, D3 = plane3
    A4, B4, C4, D4 = plane4

    A = np.array([[A1, B1, C1], [A2, B2, C2], [A3, B3, C3], [A4, B4, C4]])
    b = np.array([-D1, -D2, -D3, -D4])

    A1, A2, A3, A4 = A[:3], np.array([A[0], A[1], A[3]]), np.array([A[0], A[2], A[3]]), A[1:]
    b1, b2, b3, b4 = b[:3], np.array([b[0], b[1], b[3]]), np.array([b[0], b[2], b[3]]), b[1:]

    tip1 = np.dot(np.linalg.pinv(A1), b1)
    tip2 = np.dot(np.linalg.pinv(A2), b2)
    tip3 = np.dot(np.linalg.pinv(A3), b3)
    tip4 = np.dot(np.linalg.pinv(A4), b4)

    return (tip1+tip2+tip3+tip4)/4

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    angle_radians = np.arccos(dot_product / (norm_v1 * norm_v2))

    return angle_radians

def check_point_sides(p1, p2, p3, i, j):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    if p2[0] > p1[0] and p3[0] > p1[0]:
        if p2[1] > p1[1] and p3[1] > p1[1]:
            if p2[0] > p3[0]:
                return [i, p2], [j, p3]
            else:
                return [j, p3], [i, p2]
        if p2[1] < p1[1] and p3[1] < p1[1]:
            if p2[0] > p3[0]:
                return [j, p3], [i, p2]
            else:
                return [i, p2], [j, p3]
        if p2[1] < p1[1] and p3[1] > p1[1]:
            return [i, p2], [j, p3]
        if p2[1] > p1[1] and p3[1] < p1[1]:
            return [j, p3], [i, p2]
        
    elif p2[0] < p1[0] and p3[0] < p1[0]:
        if p2[1] > p1[1] and p3[1] > p1[1]:
            if p2[0] > p3[0]:
                return [i, p2], [j, p3]
            else:
                return [j, p3], [i, p2]
        elif p2[1] < p1[1] and p3[1] < p1[1]:    
            if p2[0] > p3[0]:
                return [j, p3], [i, p2]
            else:
                return [i, p2], [j, p3]
        elif p2[1] < p1[1] and p3[1] > p1[1]:
            return [j, p3], [i, p2]
        elif p2[1] > p1[1] and p3[1] < p1[1]:
            return [i, p2], [j, p3]
        else:
            return None, None
        
    elif p2[0] > p1[0] and p3[0] < p1[0]:
        if p2[1] > p1[1] and p3[1] > p1[1]:
            return [i, p2], [j, p3]
        elif p2[1] < p1[1] and p3[1] < p1[1]:
            return [j, p3], [i, p2]
        elif p2[1] > p1[1] and p3[1] < p1[1]:
            angle2 = angle_between_vectors(p2[:2]-p1[:2], np.array([1, 0]))
            angle3 = angle_between_vectors(p3[:2]-p1[:2], np.array([-1, 0]))
            if angle2 > angle3:
                return [i, p2], [j, p3]
            else:
                return [j, p3], [i, p2]
        elif p2[1] < p1[1] and p3[1] > p1[1]:
            angle2 = angle_between_vectors(p2[:2]-p1[:2], np.array([1, 0]))
            angle3 = angle_between_vectors(p3[:2]-p1[:2], np.array([-1, 0]))
            if angle2 > angle3:
                return [j, p3], [i, p2]
            else:
                return [i, p2], [j, p3]
        else:
            return None, None
            
    elif p2[0] < p1[0] and p3[0] > p1[0]:
        if p2[1] > p1[1] and p3[1] > p1[1]:
            return [j, p3], [i, p2]
        elif p2[1] < p1[1] and p3[1] < p1[1]:
            return [i, p2], [j, p3]
        elif p2[1] > p1[1] and p3[1] < p1[1]:
            angle2 = angle_between_vectors(p2[:2]-p1[:2], np.array([-1, 0]))
            angle3 = angle_between_vectors(p3[:2]-p1[:2], np.array([1, 0]))
            if angle2 > angle3:
                return [j, p3], [i, p2]
            else:
                return [i, p2], [j, p3]
        elif p2[1] < p1[1] and p3[1] > p1[1]:
            angle2 = angle_between_vectors(p2[:2]-p1[:2], np.array([1, 0]))
            angle3 = angle_between_vectors(p3[:2]-p1[:2], np.array([-1, 0]))
            if angle2 > angle3:
                return [i, p2], [j, p3]
            else:
                return [j, p3], [i, p2]
        else:
            return None, None
    else:
        return None, None