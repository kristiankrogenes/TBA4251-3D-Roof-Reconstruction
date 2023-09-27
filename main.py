from plotter import Plotter
from helpers import extract_variables_from_las_object, ransac, is_equal_color
import os
import laspy
import numpy as np
import alphashape 
import geopandas as gpd
import pandas as pd
from shapely import Polygon
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
                # print(self.roofs[roof_type][roof_id].point_source_id)
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
            if roof_type=="hipped":
                roofs = self.roofs_df.loc[self.roofs_df['roof_type'] == roof_type]
                rids = list(set(list(roofs['roof_id'])))
                for roof_id in rids:
                    roof_segments = roofs.loc[self.roofs_df['roof_id'] == roof_id]
                    segment_planes = list(roof_segments["plane_coefficients"])
                    if len(segment_planes) > 1:
                        for i in range(len(segment_planes)):
                            for j in range(i+1, len(segment_planes)):
                                for k in range(j+1, len(segment_planes)):
                                    A1, B1, C1 = segment_planes[i]
                                    A2, B2, C2 = segment_planes[j]
                                    A3, B3, C3 = segment_planes[k]

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
    
    def match_intersections(self, roof_id):
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

            if distance([IP[2]], [z_max]) < 0.1:
                labeled_points[i] = {}
                labeled_points[i]["segments"] = [si, sj, sk]
                labeled_points[i]["ip"] = IP
                if flat_index == 0:
                    labeled_points[i][f'{sj}{sk}'] = EPs[0]
                    labeled_points[i][f'{si}{sk}'] = EPs[1]
                if flat_index == 1:
                    labeled_points[i][f'{si}{sj}'] = EPs[0]
                    labeled_points[i][f'{si}{sk}'] = EPs[1]
                if flat_index == 2:
                    labeled_points[i][f'{si}{sj}'] = EPs[0]
                    labeled_points[i][f'{sj}{sk}'] = EPs[1]

                polygon_points.append([IP, EPs[0], EPs[1]])

        joined_intersections = list(labeled_points.keys())
        segments1, segments2 = labeled_points[joined_intersections[0]]['segments'], labeled_points[joined_intersections[1]]['segments']
        join_segments = np.array(segments1) == np.array(segments2)
        opposites_index = list(join_segments).index(False)

        if opposites_index == 0:
            sp1 = [f'{segments1[0]}{segments1[1]}', f'{segments2[0]}{segments2[1]}']
            sp2 = [f'{segments1[0]}{segments1[2]}', f'{segments2[0]}{segments2[2]}']
        if opposites_index == 1:
            sp1 = [f'{segments1[0]}{segments1[1]}', f'{segments2[0]}{segments2[1]}']
            sp2 = [f'{segments1[1]}{segments1[2]}', f'{segments2[1]}{segments2[2]}']
        if opposites_index == 2:
            sp1 = [f'{segments1[0]}{segments1[2]}', f'{segments2[0]}{segments2[2]}']
            sp2 = [f'{segments1[1]}{segments1[2]}', f'{segments2[1]}{segments2[2]}']

        polygon_points.append(
            [
                labeled_points[joined_intersections[0]]['ip'], 
                labeled_points[joined_intersections[0]][sp1[0]],
                labeled_points[joined_intersections[1]][sp1[1]],
                labeled_points[joined_intersections[1]]['ip'],
            ]
        )
        polygon_points.append(
            [
                labeled_points[joined_intersections[0]]['ip'], 
                labeled_points[joined_intersections[0]][sp2[0]],
                labeled_points[joined_intersections[1]][sp2[1]],
                labeled_points[joined_intersections[1]]['ip'],
            ]
        )
        
        return [Polygon(coords) for coords in polygon_points]


    def fit(self):

        self.roofs_df = self.fit_planes_to_roof_segments()
        self.intersection_df = self.match_planes()

        new_rows = {
            "roof_type": [],
            "roof_id": [],
            "geometry": []
        }

        roof_types = ['hipped']

        for roof_type in roof_types:
            match roof_type:
                case "hipped":
                    ids = list(set(list(self.roofs_df.loc[self.roofs_df['roof_type'] == roof_type]['roof_id'])))

                    for id in ids:
                        new_rows["roof_type"].append(roof_type)
                        new_rows["roof_id"].append(id)
                        new_rows["geometry"].append(self.match_intersections(id))
                        print(new_rows)
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