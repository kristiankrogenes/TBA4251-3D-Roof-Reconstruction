from plotter import Plotter
from helpers import extract_variables_from_las_object, ransac, is_equal_color, check_point_sides
import os
import laspy
import numpy as np
import alphashape 
import geopandas as gpd
import pandas as pd
from shapely import Polygon, MultiPoint, LineString
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

class Lo2D():

    def __init__(self, folder_path):
        self.roofs = self.fetch_data_from_point_cloud(folder_path)

    def fetch_data_from_point_cloud(self, folder_path):
        DATA_FOLDERS = os.listdir(folder_path)

        data = {}

        for data_folder in DATA_FOLDERS:
            if data_folder != 'roof categories.jpg':
                data[data_folder] = {}
                FILES_PATH = f'{folder_path}/{data_folder}/'
                file_names = os.listdir(FILES_PATH)

                for file_name in file_names:
                    FILE_PATH = f'{FILES_PATH}/{file_name}'
                    laz = laspy.read(FILE_PATH)
                    las = laspy.convert(laz)
                    # las.write(f'./data/las_roofs/{data_folder}/{file_name[:-4]}.las')
                    data[data_folder][file_name[:-4]] = las
        data = self.classify_roof_segments(data)
        return data 
    
    def classify_roof_segments(self, data):
        for roof_type in data:
            for roof_id in data[roof_type]:
                roof = data[roof_type][roof_id]
                rgb = [(r, g, b) for r, g, b in zip(roof.red, roof.green, roof.blue)]
                unique_rgbs = list(set(rgb))
                segments = np.array([None for _ in range(len(rgb))])
                for i, unique_rgb in enumerate(unique_rgbs):
                    s = [is_equal_color(rgbi, unique_rgb) for rgbi in rgb]
                    segments = [i if k else j for k, j in zip(s, segments)]
                data[roof_type][roof_id].point_source_id = segments
        return data
    # Blue, lihtgreen, red, lightblue, pink, green

    # [3,    0,      1,        5,     4,   2]
    # [lb, blue, lightgreen, green, pink, red]



    # (3, 2, 5), (0, 1, 4), (4, 5)

    def fit_planes_to_roof_segments(self):
        roofs_df = pd.DataFrame(columns=[
            'roof_type', 
            'roof_id', 
            'segment_id', 
            'x_coordinates', 
            'y_coordinates', 
            'z_coordinates', 
            'plane_coefficients'
        ])

        for roof_type in self.roofs:
            for roof_id in self.roofs[roof_type]:
                segment_ids = list(set(self.roofs[roof_type][roof_id].point_source_id))
                min_z = np.inf
                for sid in segment_ids:
                    roof_segment_indexes = np.array([id==sid for id in self.roofs[roof_type][roof_id].point_source_id])
                    roof_segment_coords = np.array(self.roofs[roof_type][roof_id].xyz)[roof_segment_indexes]

                    x, y, z = roof_segment_coords[:, 0], roof_segment_coords[:, 1], roof_segment_coords[:, 2]

                    min_z = min(z) if min(z) < min_z else min_z

                    # Create a matrix A and vector b for the LSM equation Ax = b
                    A = np.column_stack((x, y, np.ones(len(roof_segment_coords)) ))
                    b = z

                    # Use LSM to find the plane coefficients (a, b, c, d)
                    coeffs, _, _, _ = lstsq(A, b)

                    new_row = {
                        'roof_type': [roof_type],
                        'roof_id': [roof_id],
                        'segment_id': [sid],
                        'x_coordinates': [x],
                        'y_coordinates': [y],
                        'z_coordinates': [z],
                        'plane_coefficients': [coeffs]
                    }
                    df = pd.DataFrame(new_row)
                    roofs_df = pd.concat([roofs_df, df])

        return roofs_df.reset_index()

    def match_planes(self):
        roof_ids = list(set(list(self.roofs_df['roof_id'])))

        roof_types = list(set(list(self.roofs_df['roof_type'])))
        intersection_df = pd.DataFrame(columns=["roof_id", "intersection_point", "triple_intersection", "direction_vector", "s1", "s2"])

        for roof_type in roof_types:
            if roof_type=="hipped" or "corner_element":
                roofs = self.roofs_df.loc[self.roofs_df['roof_type'] == roof_type]
                rids = list(set(list(roofs['roof_id'])))
                for roof_id in rids:
                    roof_segments = roofs.loc[self.roofs_df['roof_id'] == roof_id]
                    segment_ids = list(roof_segments["segment_id"])
                    segment_planes = list(roof_segments["plane_coefficients"])
                    if len(segment_planes) > 1:
                        for indexi, i in enumerate(segment_ids):
                            for indexj, j in enumerate(segment_ids[indexi+1:]):
                                for indexk, k in enumerate(segment_ids[indexi+1+indexj+1:]):
                                    A1, B1, C1 = segment_planes[indexi]
                                    A2, B2, C2 = segment_planes[indexi+1+indexj]
                                    A3, B3, C3 = segment_planes[indexi+1+indexj+1+indexk]

                                    A = np.array([[A1, B1, -1], [A2, B2, -1], [A3, B3, -1]])
                                    b = np.array([-C1, -C2, -C3])

                                    triple_intersection = np.dot(np.linalg.inv(A), b)

                                    # Find intersection points where z=0
                                    A1, A2, A3 = A[:2,:2], A[1:,:2], np.array([A[0][:2], A[2][:2]])
                                    b1, b2, b3 = b[:2], b[1:], np.array([b[0], b[2]])

                                    ip1 = np.dot(np.linalg.inv(A1), b1)
                                    ip2 = np.dot(np.linalg.inv(A2), b2)
                                    ip3 = np.dot(np.linalg.inv(A3), b3)

                                    dv1 = np.cross(A[0], A[1])
                                    dv2 = np.cross(A[1], A[2])
                                    dv3 = np.cross(A[0], A[2])

                                    new_row = {
                                        "roof_id": [roof_id], 
                                        "intersection_point": [[np.append(ip1, 0, axis=None), np.append(ip2, 0, axis=None), np.append(ip3, 0, axis=None)]], 
                                        "triple_intersection": [triple_intersection],
                                        "direction_vector": [[dv1, dv2, dv3]], 
                                        "s1": [i], 
                                        "s2": [j], 
                                        "s3": [k], 
                                    }
                                    intersection_df = pd.concat([intersection_df, pd.DataFrame(new_row)])
        return intersection_df.reset_index()
    
    def create_roof_polygons_from_planes(self, roof_id):
        polygon_points = []
        intersections = self.intersection_df.loc[self.intersection_df['roof_id'] == roof_id]
        intersections = intersections.reset_index()

        meta_roof = self.roofs_df.loc[self.roofs_df['roof_id'] == roof_id]
        z_min = min([min(z_coords) for z_coords in list(meta_roof["z_coordinates"])])
        z_max = max([max(z_coords) for z_coords in list(meta_roof["z_coordinates"])])

        labeled_points = {}

        for i in range(len(intersections)):
            IP = intersections.loc[i]['triple_intersection']
            points = list(intersections.loc[i]['intersection_point'])
            dvs = list(intersections.loc[i]['direction_vector'])
        
            EPs = [p + z_min/dv[2] * dv for p, dv in zip(points, dvs) if abs(dv[2]) > 0.1]

            distance = lambda p1, p2: np.sqrt(np.sum([(j-i)**2 for i, j in zip(p1, p2)]))

            flat_dv = [False if abs(dv[2]) > 0.1 else True for dv in dvs]
            flat_index = flat_dv.index(True)

            si, sj, sk = int(intersections.loc[i]["s1"]), int(intersections.loc[i]["s2"]), int(intersections.loc[i]["s3"])


            if distance([IP[2]], [z_max]) < 0.2:
                labeled_points[i] = {}
                labeled_points[i]["segments"] = [si, sj, sk]
                labeled_points[i]["ip"] = IP
                if flat_index == 0:
                    labeled_points[i][f'{sj}{sk}'] = EPs[0]
                    labeled_points[i][f'{si}{sk}'] = EPs[1]
                    left, right = check_point_sides(IP, EPs[0], EPs[1], sj, si)
                    labeled_points[i]['neighbours'] = [left[0], sk, right[0]]
                    labeled_points[i]['right_ep'] = right[1]
                    labeled_points[i]['left_ep'] = left[1]
                    labeled_points[i]['center'] = [sj, sk, si]
                    labeled_points[i]['p1'] = EPs[0]
                    labeled_points[i]['p2'] = EPs[1]
                if flat_index == 1:
                    labeled_points[i][f'{si}{sj}'] = EPs[0]
                    labeled_points[i][f'{si}{sk}'] = EPs[1]
                    left, right = check_point_sides(IP, EPs[0], EPs[1], sj, sk)
                    labeled_points[i]['neighbours'] = [left[0], si, right[0]]
                    labeled_points[i]['right_ep'] = right[1]
                    labeled_points[i]['left_ep'] = left[1]
                    labeled_points[i]['center'] = [sj, si, sk]
                    labeled_points[i]['p1'] = EPs[0]
                    labeled_points[i]['p2'] = EPs[1]
                if flat_index == 2:
                    labeled_points[i][f'{si}{sj}'] = EPs[0]
                    labeled_points[i][f'{sj}{sk}'] = EPs[1]
                    left, right = check_point_sides(IP, EPs[0], EPs[1], si, sk)
                    labeled_points[i]['neighbours'] = [left[0], sj, right[0]]
                    labeled_points[i]['right_ep'] = right[1]
                    labeled_points[i]['left_ep'] = left[1]
                    labeled_points[i]['center'] = [si, sj, sk]
                    labeled_points[i]['p1'] = EPs[0]
                    labeled_points[i]['p2'] = EPs[1]

        joined_intersections = list(labeled_points.keys())

        neighbours = [labeled_points[i]['neighbours'] for i in joined_intersections]
        counts = [[neighbours[c][1]==nb[0] or neighbours[c][1]==nb[2] for nb in neighbours] for c in range(len(neighbours))]
        counts2 = [n for n, count in enumerate(counts) if not any(count)]

        for ei, edge_index in enumerate(counts2):

            joined_intersection = labeled_points[joined_intersections[edge_index]]
            
            other_ei = 0 if ei==1 else 1
            other_ji = labeled_points[joined_intersections[counts2[other_ei]]]
            other_right_neighbour = other_ji['neighbours'][2]
            other_left_neighbour = other_ji['neighbours'][0]

            if len(neighbours) > 2:
                left_neighbour, right_neighbour = joined_intersection['neighbours'][0], joined_intersection['neighbours'][2]

                left_index = [neighbour[1]==left_neighbour for neighbour in neighbours].index(True)
                right_index = [neighbour[1]==right_neighbour for neighbour in neighbours].index(True)

                left_intersection = labeled_points[joined_intersections[left_index]]
                left_intersection_left_neighbour_index = left_intersection['center'].index(other_right_neighbour)
                
                right_intersection = labeled_points[joined_intersections[right_index]]
                right_intersection_right_neighbour_index = right_intersection['center'].index(other_left_neighbour)
                
                polygon_points.append(
                    [
                        joined_intersection['ip'], 
                        joined_intersection['right_ep'],
                        joined_intersection['left_ep'],
                    ]
                )
                polygon_points.append(
                    [
                        joined_intersection['ip'], 
                        joined_intersection['left_ep'],
                        left_intersection['ip'],
                        left_intersection['p1'] if left_intersection_left_neighbour_index==0 else left_intersection['p2'],
                    ]
                )
                polygon_points.append(
                    [
                        joined_intersection['ip'],
                        joined_intersection['right_ep'], 
                        right_intersection['ip'],
                        right_intersection['p1'] if right_intersection_right_neighbour_index==0 else right_intersection['p2'],
                    ]
                )
            else:
                polygon_points.append(
                    [
                        joined_intersection['ip'], 
                        joined_intersection['right_ep'],
                        joined_intersection['left_ep'],
                    ]
                )
                polygon_points.append(
                    [
                        joined_intersection['ip'], 
                        joined_intersection['left_ep'],
                        other_ji['ip'],
                        other_ji['right_ep']

                    ]
                )

        """
        print(roof_id)
        print(segments)
        print()
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):

                s1, s2 = set(segments[i]), set(segments[j])
                equals = list(s1.intersection(s2))
                opposites = list(s1.symmetric_difference(s2))
                
                if len(equals) == 2:
                    roof_info = self.roofs_df.loc[self.roofs_df['roof_id'] == roof_id]
                    sdv1 = list(list(roof_info.loc[roof_info['segment_id'] == opposites[0]]['plane_coefficients'])[0][:2])+[-1]
                    sdv2 = list(list(roof_info.loc[roof_info['segment_id'] == opposites[1]]['plane_coefficients'])[0][:2])+[-1]
                    cross_product = np.cross(sdv1, sdv2)

                    if abs(cross_product[0]) > 0.1 and abs(cross_product[1]) > 0.1:
                        if abs(cross_product[2] < 0.1):

                            print(segments[i])
                            print(segments[j])
                            
                            sp1 = [sorted([opposites[0], equals[0]]), sorted([opposites[1], equals[0]])]
                            sp2 = [sorted([opposites[0], equals[1]]), sorted([opposites[1], equals[1]])]
                            sp1 = [f'{str(sp[0])}{str(sp[1])}' for sp in sp1]
                            sp2 = [f'{str(sp[0])}{str(sp[1])}' for sp in sp2]

                            try:
                                polygon_points.append(
                                    [
                                        labeled_points[joined_intersections[i]]['ip'], 
                                        labeled_points[joined_intersections[i]][sp1[0]],
                                        labeled_points[joined_intersections[j]][sp1[1]],
                                        labeled_points[joined_intersections[j]]['ip'],
                                    ]
                                )
                            except Exception as ex:
                                print("First", ex)
                            
                            try:
                                polygon_points.append(
                                    [
                                        labeled_points[joined_intersections[i]]['ip'], 
                                        labeled_points[joined_intersections[i]][sp2[0]],
                                        labeled_points[joined_intersections[j]][sp2[1]],
                                        labeled_points[joined_intersections[j]]['ip'],
                                    ]
                                )
                            except Exception as ex:
                                print("Second", ex)
                            print()
        """
        return [MultiPoint(coords).convex_hull for coords in polygon_points]

    def fit(self):

        self.roofs_df = self.fit_planes_to_roof_segments()
        self.intersection_df = self.match_planes()

        new_rows = {
            "roof_type": [],
            "roof_id": [],
            "geometry": []
        }

        roof_types = ['hipped', 'corner_element']

        for roof_type in roof_types:
            ids = list(set(list(self.roofs_df.loc[self.roofs_df['roof_type'] == roof_type]['roof_id'])))

            for id in ids:
                new_rows["roof_type"].append(roof_type)
                new_rows["roof_id"].append(id)

                match roof_type:
                    case "corner_element":
                        new_rows["geometry"].append(self.create_roof_polygons_from_planes(id))
                    case "hipped":
                        new_rows["geometry"].append(self.create_roof_polygons_from_planes(id))
                    



        self.polygon_roofs_df = pd.DataFrame(new_rows)
    
    def plot_data(self, plot_type):
        match plot_type:
            case "pointcloud_scatterplot_3d":
                for roof_type in self.roofs:
                    for roof_id in self.roofs[roof_type]:
                        plot_title = f'{roof_type}-{roof_id}'
                        Plotter.scatterplot_3d(plot_title, *extract_variables_from_las_object(self.roofs[roof_type][roof_id]))
            case "segmented_roof_polygons_2d":
                roof_ids = list(set(list(self.polygon_df['roof_id'])))
                for id in roof_ids:
                    roof_segments = self.polygon_df.loc[self.polygon_df['roof_id'] == id]
                    polygons = [poly for poly in list(roof_segments['geometry'])]
                    title=list(roof_segments['roof_type'])[0]+"-"+id
                    Plotter.segmented_polygons_2d(title, polygons)
            case "clustered_segmented_roof_polygons_2d":
                roof_ids = list(set(list(self.df_clustered_roof_polygons['roof_id'])))
                for id in roof_ids:
                    roof_segments = self.df_clustered_roof_polygons.loc[self.df_clustered_roof_polygons['roof_id'] == id]
                    polygons = [poly for poly in list(roof_segments['geometry'])]
                    Plotter.segmented_polygons_2d(id, polygons)
            case "double_segmented_polygons_2d":
                roof_ids = list(set(list(self.df_clustered_roof_polygons['roof_id'])))
                for id in roof_ids:
                    roof_segments1 = self.polygon_df.loc[self.polygon_df['roof_id'] == id]
                    roof_segments2 = self.df_clustered_roof_polygons.loc[self.df_clustered_roof_polygons['roof_id'] == id]
                    polygons1 = [poly for poly in list(roof_segments1['geometry'])]
                    polygons2 = [poly for poly in list(roof_segments2['geometry'])]
                    Plotter.double_segmented_polygons_2d(id, polygons1, polygons2)
            case "double_segmented_polygons_3d":
                roof_ids = list(set(list(self.df_clustered_roof_polygons['roof_id'])))
                for id in roof_ids:
                    roof_segments1 = self.polygon_df.loc[self.polygon_df['roof_id'] == id]
                    roof_segments2 = self.df_clustered_roof_polygons.loc[self.df_clustered_roof_polygons['roof_id'] == id]
                    polygons1 = [poly for poly in list(roof_segments1['geometry'])]
                    polygons2 = [poly for poly in list(roof_segments2['geometry'])]
                    Plotter.double_segmented_polygons_3d(id, polygons1, polygons2)
            case "planes_3d":
                roof_ids = list(set(list(self.roofs_df['roof_id'])))
                for id in roof_ids:
                    roof_segments = self.roofs_df.loc[self.roofs_df['roof_id'] == id]
                    # roof_segments2 = self.df_clustered_roof_polygons.loc[self.df_clustered_roof_polygons['roof_id'] == id]
                    planes = [plane for plane in list(roof_segments['plane_coefficients'])]
                    coords = [[x, y, z] for x, y, z in zip(list(roof_segments['x_coordinates']), list(roof_segments['y_coordinates']), list(roof_segments['z_coordinates']))]
                    # polygons2 = [poly for poly in list(roof_segments2['geometry'])]
                    Plotter.planes_3d(id, planes, coords)
            case "intersection_lines_3d":
                roof_ids = list(set(list(self.intersection_df['roof_id'])))
                for roof_id in roof_ids:
                    roof_segments = self.roofs_df.loc[self.roofs_df['roof_id'] == roof_id]
                    x_coords = np.array([[min(x), max(x)] for x in list(roof_segments['x_coordinates'])])
                    xmin, xmax = min(x_coords[:, 0]), max(x_coords[:, 1])
                    y_coords = np.array([[min(y), max(y)] for y in list(roof_segments['y_coordinates'])])
                    ymin, ymax = min(y_coords[:, 0]), max(y_coords[:, 1])
                    z_coords = np.array([[min(z), max(z)] for z in list(roof_segments['z_coordinates'])])
                    zmin, zmax = min(z_coords[:, 0]), max(z_coords[:, 1])

                    intersections = self.intersection_df.loc[self.intersection_df['roof_id'] == roof_id]
                    intersection_points = list(intersections["intersection_point"])
                    triple_intersections = list(intersections["triple_intersection"])
                    direction_vectors = list(intersections["direction_vector"])

                    ts = [(zmin/dv[2], zmax/dv[2]) for dv in direction_vectors]
                    points = [[(ip[0]+t[0]*dv[0], ip[1]+t[0]*dv[1], t[0]*dv[2]), (ip[0]+t[1]*dv[0], ip[1]+t[1]*dv[1], t[1]*dv[2])] for ip, dv, t in zip(intersection_points, direction_vectors, ts)]

                    Plotter.intersection_lines_3d(points, xmin, xmax, ymin, ymax, zmin, zmax, triple_intersections)
            case "roof_polygons_3d":
                roof_ids = list(set(list(self.polygon_roofs_df['roof_id'])))
                for id in roof_ids:
                    roof = self.polygon_roofs_df.loc[self.polygon_roofs_df['roof_id'] == id]
                    polygons = list(roof['geometry'])[0]
                    Plotter.double_segmented_polygons_3d(id, polygons, polygons)



if __name__ == "__main__":

    DATA_FOLDER_PATH = './data/roofs'

    lo2d = Lo2D(DATA_FOLDER_PATH)
    lo2d.fit()

    # lo2d.plot_data("pointcloud_scatterplot_3d")
    # lo2d.plot_data("segmented_roof_polygons_2d")
    # lo2d.plot_data("double_segmented_polygons_2d")
    # lo2d.plot_data("double_segmented_polygons_3d")
    # lo2d.plot_data("planes_3d")
    # lo2d.plot_data("intersection_lines_3d")
    lo2d.plot_data("roof_polygons_3d")