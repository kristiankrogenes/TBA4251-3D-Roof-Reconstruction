import os
import laspy
import numpy as np
import alphashape 
import geopandas as gpd
import pandas as pd
from scipy import optimize
import math
from sklearn.cluster import DBSCAN
from shapely import Polygon

def plane_equation(params, x, y, z):
    A, B, C, D = params
    return A * x + B * y + C * z + D

def plane_parameters(points):

    V1 = (points[1][0] - points[0][0], points[1][1] - points[0][1], points[1][2] - points[0][2])
    V2 = (points[2][0] - points[0][0], points[2][1] - points[0][1], points[2][2] - points[0][2])

    A = np.float32(V1[1] * V2[2] - V1[2] * V2[1])
    B = np.float32(V1[2] * V2[0] - V1[0] * V2[2])
    C = np.float32(V1[0] * V2[1] - V1[1] * V2[0])

    D = -A * np.float32(points[0][0]) - B * np.float32(points[0][1]) - C * np.float32(points[0][2])
    
    return [A, B, C, D]

def point_to_plane_distance(params, point):
    x, y, z = point
    return np.abs(plane_equation(params, x, y, z)) / np.sqrt(params[0]**2 + params[1]**2 + params[2]**2)


def ransac(x, y, z, s):
    points = np.array([[xi, yi, zi] for xi, yi, zi in zip(x, y, z)])

    num_iterations = 100
    sample_size = 3
    threshold_distance = 1

    best_params = None
    best_inliers = None
    best_num_inliers = 0

    for _ in range(num_iterations):

        sample_indices = np.random.choice(len(points), size=sample_size, replace=False)
        # sample_indices = np.array([10, 20, 30])

        sample_points = points[sample_indices]

        # Fit a plane to the sampled points using least squares
        # params, _ = optimize.curve_fit(plane_equation, sample_points[0], sample_points[1], sample_points[2])
        params = plane_parameters(sample_points)

        # Compute distances from all points to the fitted plane
        distances = np.array([point_to_plane_distance(params, point) for point in points])

        # Count inliers (points within the threshold distance of the plane)
        num_inliers = np.sum(distances < threshold_distance)

        # Check if this is the best model so far
        if num_inliers >= best_num_inliers:
            best_num_inliers = num_inliers
            best_params = params
            best_inliers = points[distances < threshold_distance]
    
    print("Best plane parameters (A, B, C, D):", best_params)
    print("Number of inliers:", best_num_inliers)
    print("Inlier points:", best_inliers)

    return best_params, best_inliers

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

def create_polygon_from_roof_segment(las):
    poly = alphashape.alphashape(las.xyz, 1.1)
    poly_df = gpd.GeoDataFrame([{'geometry': poly, 'classification': 1}, {'geometry': poly, 'classification': 2}])
    return poly_df

def get_alpha_shape_polygons(roof_segments):
    """
    Generates polygons from point cloud data with Alpha Shape algorithm \n
    Returns Pandas DataFrame with all segments in a roof
    """
    alpha_shape = [alphashape.alphashape(l.xyz, 0) for l in roof_segments]
    df_list = [{'geometry': alpha_shape[i], 'classification': i} for i in range(len(alpha_shape))]
    df = gpd.GeoDataFrame(df_list)
    thresh = lambda x: 0.7 if x.area > 10 else 0.4
    df.geometry = df.geometry.apply(lambda x: x.simplify(thresh(x), preserve_topology=False))
    return df

def cluster_roof_polygons(self):

    df_clustered_roof_polygons = pd.DataFrame(columns=['roof_id', 'geometry'])
    roof_ids = list(set(list(self.polygon_df['roof_id'])))

    for roof_id in roof_ids:

        roof_segments_df = self.polygon_df.loc[self.polygon_df['roof_id'] == roof_id]

        roof_segment_coords = [[coord[:3] for coord in roof_segments_df.iloc[i].geometry.exterior.coords] for i in range(len(roof_segments_df))]

        roof_coords = np.concatenate(roof_segment_coords, axis=0)

        clustering = DBSCAN(eps=2, min_samples=2).fit(roof_coords)

        clusters = {}
        updated_clustering_labels = []
        for i, label in enumerate(clustering.labels_):
            if label==-1:
                new_label = label*i-1
                clusters[new_label] = {i: roof_coords[i]}
                updated_clustering_labels.append(new_label)
            else:
                if label in clusters.keys():
                    clusters[label][i] = roof_coords[i]
                else:
                    clusters[label] = {i: roof_coords[i]}
                updated_clustering_labels.append(label)

        mean_cluster_coords = {}
        for c in clusters:
            coords = [clusters[c][p] for p in clusters[c]]
            mean_coords_x = np.sum([coords[i][0] for i in range(len(coords))]) / len(coords)
            mean_coords_y = np.sum([coords[i][1] for i in range(len(coords))]) / len(coords)
            mean_coords_z = np.sum([coords[i][2] for i in range(len(coords))]) / len(coords)
            mean_cluster_coords[c] = (mean_coords_x, mean_coords_y, mean_coords_z)

        updated_coords = [mean_cluster_coords[c] for c in updated_clustering_labels]

        inndeling = [len(coords) for coords in roof_segment_coords]
        updated_segment_coords = []
        index_memory = 0
        for i in range(len(inndeling)):
            updated_segment_coords.append(updated_coords[index_memory:index_memory+inndeling[i]])
            index_memory += inndeling[i]

        clustered_df = {
            'roof_id': [roof_id for _ in range(len(roof_segments_df))], 
            'geometry': [Polygon(coords) for coords in updated_segment_coords]
        }
        df = pd.DataFrame(clustered_df)
        df_clustered_roof_polygons = pd.concat([df_clustered_roof_polygons, df])

    return df_clustered_roof_polygons.reset_index()

def fit_polygons_to_roofs(self):
    polygon_df = pd.DataFrame(columns=['roof_type', 'roof_id', 'geometry'])

    for roof_type in self.roofs:
        for roof_id in self.roofs[roof_type]:
            roof = self.roofs[roof_type][roof_id]
            segment_ids = list(set(roof.point_source_id))

            for sid in segment_ids:
                roof_segment_coords = np.array([id==sid for id in roof.point_source_id])
                coords = np.array(roof.xyz)[roof_segment_coords]
                alpha = alphashape.alphashape(coords, 0)
                pdf = {"roof_type": [roof_type], "roof_id": [roof_id], "geometry": [alpha]}
                df = pd.DataFrame(pdf)
                polygon_df = pd.concat([polygon_df, df])

    thresh = lambda x: 0.7 if x.area > 10 else 0.4
    polygon_df.geometry = polygon_df.geometry.apply(lambda x: x.simplify(thresh(x), preserve_topology=False))

    return polygon_df.reset_index()

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
    
p1 = [5.67309600e+05, 7.02657393e+06, 1.69043021e+02]
p2, p3 = [5.67313312e+05, 7.02657969e+06, 1.67060000e+02], [5.67315358e+05, 7.02657018e+06, 1.67060000e+02]
a = check_point_sides(p1, p2, p3, 0, 4)
print(p2[0] > p1[0], p3[0] > p1[0])
print(p2[1] > p1[1], p3[1] < p1[1])
print(a)