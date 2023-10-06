from plotter import Plotter
from helpers import extract_variables_from_las_object, is_equal_color, check_point_sides, intersect_3planes, intersect_2planes, find_min_max_values
import os
import laspy
import numpy as np
import pandas as pd
from shapely import Polygon, MultiPoint, LineString
from scipy.linalg import lstsq

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
                    coeffs = [coeffs[0], coeffs[1], -1, coeffs[2]]

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
            
            roofs = self.roofs_df.loc[self.roofs_df['roof_type'] == roof_type]
            rids = list(set(list(roofs['roof_id'])))
            for roof_id in rids:
                roof_segments = roofs.loc[self.roofs_df['roof_id'] == roof_id]
                segment_ids = list(roof_segments["segment_id"])
                segment_planes = list(roof_segments["plane_coefficients"])

                if roof_type=="hipped" or "corner_element":
                    for indexi, i in enumerate(segment_ids):
                        for indexj, j in enumerate(segment_ids[indexi+1:]):
                            for indexk, k in enumerate(segment_ids[indexi+1+indexj+1:]):
                                tip, ips, dvs = intersect_3planes(
                                    segment_planes[indexi], 
                                    segment_planes[indexi+1+indexj], 
                                    segment_planes[indexi+1+indexj+1+indexk]
                                )
                                new_row = {
                                    "roof_id": [roof_id], 
                                    "intersection_point": [ips], 
                                    "triple_intersection": [tip],
                                    "direction_vector": [dvs], 
                                    "s1": [i], 
                                    "s2": [j], 
                                    "s3": [k], 
                                }
                                intersection_df = pd.concat([intersection_df, pd.DataFrame(new_row)])
                if roof_type=="gabled":
                    for indexi, i in enumerate(segment_ids):
                        for indexj, j in enumerate(segment_ids[indexi+1:]):
                                intersection, direction_vector = intersect_2planes(
                                    segment_planes[indexi], 
                                    segment_planes[indexi+1+indexj], 
                                )
                                new_row = {
                                    "roof_id": [roof_id], 
                                    "intersection_point": [intersection], 
                                    "triple_intersection": [None],
                                    "direction_vector": [direction_vector], 
                                    "s1": [i], 
                                    "s2": [j], 
                                }
                                intersection_df = pd.concat([intersection_df, pd.DataFrame(new_row)])

        return intersection_df.reset_index()
    
    def sort_flat_roof_segment(self, roof_type, roof_id):
        roof = self.roofs_df.loc[self.roofs_df['roof_id'] == roof_id]
        roof_plane = list(roof["plane_coefficients"])[0]
        dv1 = roof_plane[:3]
        dv2 = np.cross(dv1, [0, 0, 1])

        x, y, z = list(roof["x_coordinates"])[0], list(roof["y_coordinates"])[0], list(roof["z_coordinates"])[0]
        p1, p2, p3, p4, _, _ = find_min_max_values(x, y, z)

        edge_plane_1 = [dv1[0], dv1[1], 0, -dv1[0]*p3[0]-dv1[1]*p3[1]-0*p3[2]]
        edge_plane_2 = [dv1[0], dv1[1], 0, -dv1[0]*p4[0]-dv1[1]*p4[1]-0*p4[2]]
        edge_plane_3 = [dv2[0], dv2[1], 0, -dv2[0]*p1[0]-dv2[1]*p1[1]-0*p1[2]]
        edge_plane_4 = [dv2[0], dv2[1], 0, -dv2[0]*p2[0]-dv2[1]*p2[1]-0*p2[2]]

        tip1, _, _ = intersect_3planes(roof_plane, edge_plane_1, edge_plane_3)
        tip2, _, _ = intersect_3planes(roof_plane, edge_plane_1, edge_plane_4)
        tip3, _, _ = intersect_3planes(roof_plane, edge_plane_2, edge_plane_3)
        tip4, _, _ = intersect_3planes(roof_plane, edge_plane_2, edge_plane_4)

        labeled_points = {"p1": [tip1, tip2, tip3, tip4]}

        return self.create_polygons(roof_type, labeled_points)

    def sort_gabled_roof_segments(self, roof_type, roof_id):
        intersections = self.intersection_df.loc[self.intersection_df['roof_id'] == roof_id]
        intersections = intersections.reset_index()

        roof = self.roofs_df.loc[self.roofs_df['roof_id'] == roof_id]

        x1, y1, z1 = list(roof["x_coordinates"])[0], list(roof["y_coordinates"])[0], list(roof["z_coordinates"])[0]
        x2, y2, z2 = list(roof["x_coordinates"])[1], list(roof["y_coordinates"])[1], list(roof["z_coordinates"])[1]

        s1p1, s1p2, _, _, s1p5, _ = find_min_max_values(x1, y1, z1)
        _, _, _, _, s2p5, _ = find_min_max_values(x2, y2, z2)

        labeled_points = {}

        plane1, plane2 = list(roof["plane_coefficients"])[0], list(roof["plane_coefficients"])[1]

        direction_vector = intersections.loc[0]["direction_vector"]
        
        edge_plane1 = [direction_vector[0], direction_vector[1], 0, -direction_vector[0]*s1p1[0] - direction_vector[1]*s1p1[1] - 0*s1p1[2]]
        edge_plane2 = [direction_vector[0], direction_vector[1], 0, -direction_vector[0]*s1p2[0] - direction_vector[1]*s1p2[1] - 0*s1p2[2]]

        tip1, ips1, dvs1 = intersect_3planes(plane1, plane2, edge_plane1)
        tip2, ips2, dvs2 = intersect_3planes(plane1, plane2, edge_plane2)

        EPs1p1 = ips1[2]+s1p5[2]/dvs1[2][2] * dvs1[2]
        EPs1p2 = ips2[2]+s1p5[2]/dvs2[2][2] * dvs2[2]
        EPs2p1 = ips1[1]+s2p5[2]/dvs1[1][2] * dvs1[1]
        EPs2p2 = ips2[1]+s2p5[2]/dvs2[1][2] * dvs2[1]

        labeled_points["p1"] = [tip1, tip2, EPs1p1, EPs1p2]
        labeled_points["p2"] = [tip1, tip2, EPs2p1, EPs2p2]

        return self.create_polygons(roof_type, labeled_points)

    def sort_telement_roof_segments(self, roof_type, roof_id):

        roof = self.roofs_df.loc[self.roofs_df['roof_id'] == roof_id]
        segment_ids = list(roof["segment_id"])
        segment_planes = list(roof["plane_coefficients"])

        xs, ys, zs = list(roof["x_coordinates"]), list(roof["y_coordinates"]), list(roof["z_coordinates"])

        intersections = []
        gabled_matches = []
        for indexi, i in enumerate(segment_ids):
            for indexj, j in enumerate(segment_ids[indexi+1:]):
                    intersection, direction_vector = intersect_2planes(
                        segment_planes[indexi], 
                        segment_planes[indexi+1+indexj], 
                    )
                    if abs(direction_vector[2]) < 0.1:
                        gabled_matches.append([indexi,  indexi+1+indexj])
                        intersections.append([intersection, direction_vector])

        main_roof_match = 0 if max(zs[gabled_matches[0][0]]) > max(zs[gabled_matches[1][0]]) else 1
        sub_roof_match = 1 if main_roof_match == 0 else 0

        minx_sub_seg0, maxx_sub_seg0, _, _, minz_sub_seg0, _ = find_min_max_values(xs[gabled_matches[sub_roof_match][0]], ys[gabled_matches[sub_roof_match][0]], zs[gabled_matches[sub_roof_match][0]])
        _, _, _, _, minz_sub_seg1, _ = find_min_max_values(xs[gabled_matches[sub_roof_match][1]], ys[gabled_matches[sub_roof_match][1]], zs[gabled_matches[sub_roof_match][1]])
        minx_main_seg0, maxx_main_seg0, _, _, minz_main_seg0, _ = find_min_max_values(xs[gabled_matches[main_roof_match][0]], ys[gabled_matches[main_roof_match][0]], zs[gabled_matches[main_roof_match][0]])
        _, _, _, _, minz_main_seg1, _ = find_min_max_values(xs[gabled_matches[main_roof_match][1]], ys[gabled_matches[main_roof_match][1]], zs[gabled_matches[main_roof_match][1]])
        
        global_zmin = min(minz_sub_seg0[2], minz_sub_seg1[2], minz_main_seg0[2], minz_main_seg1[2])

        distance = lambda p1, p2: np.sqrt(np.sum([(j-i)**2 for i, j in zip(p1, p2)]))
        closest_main_segment = 0 if distance(minz_sub_seg0, minz_main_seg0) < distance(minz_sub_seg0, minz_main_seg1) else 1

        main_sub_tip, main_sub_ips, main_sub_dvs = intersect_3planes(
            segment_planes[gabled_matches[sub_roof_match][0]], 
            segment_planes[gabled_matches[sub_roof_match][1]], 
            segment_planes[gabled_matches[main_roof_match][closest_main_segment]]
        )
        main_sub_eps = [p + global_zmin/dv[2] * dv for p, dv in zip(main_sub_ips, main_sub_dvs) if abs(dv[2]) > 0.1]

        sub_edge_ref_point = minx_sub_seg0 if distance(minx_sub_seg0, main_sub_tip) > distance(maxx_sub_seg0, main_sub_tip) else maxx_sub_seg0
        sub_intersection_dv = intersections[sub_roof_match][1]
        sub_edge_plane = [sub_intersection_dv[0], sub_intersection_dv[1], 0, -sub_intersection_dv[0]*sub_edge_ref_point[0] - sub_intersection_dv[1]*sub_edge_ref_point[1] - 0*sub_edge_ref_point[2]]
        
        sub_tip, sub_ips, sub_dvs = intersect_3planes(
            segment_planes[gabled_matches[sub_roof_match][0]], 
            segment_planes[gabled_matches[sub_roof_match][1]], 
            sub_edge_plane
        )
        sub_eps = [p + global_zmin/dv[2] * dv for p, dv in zip(sub_ips, sub_dvs) if abs(dv[2]) > 0.1]

        main_intersection_dv = intersections[main_roof_match][1]
        main_edge_plane1 = [main_intersection_dv[0], main_intersection_dv[1], 0, -main_intersection_dv[0]*minx_main_seg0[0] - main_intersection_dv[1]*minx_main_seg0[1] - 0*minx_main_seg0[2]]
        main_edge_plane2 = [main_intersection_dv[0], main_intersection_dv[1], 0, -main_intersection_dv[0]*maxx_main_seg0[0] - main_intersection_dv[1]*maxx_main_seg0[1] - 0*maxx_main_seg0[2]]

        main_tip1, main_ips1, main_dvs1 = intersect_3planes(
            segment_planes[gabled_matches[main_roof_match][0]], 
            segment_planes[gabled_matches[main_roof_match][1]], 
            main_edge_plane1
        )
        main_eps1 = [p + global_zmin/dv[2] * dv for p, dv in zip(main_ips1, main_dvs1) if abs(dv[2]) > 0.1]

        main_tip2, main_ips2, main_dvs2 = intersect_3planes(
            segment_planes[gabled_matches[main_roof_match][0]], 
            segment_planes[gabled_matches[main_roof_match][1]], 
            main_edge_plane2
        )
        main_eps2 = [p + global_zmin/dv[2] * dv for p, dv in zip(main_ips2, main_dvs2) if abs(dv[2]) > 0.1]

        ind = 1 if closest_main_segment==0 else 1
        return self.create_polygons(roof_type, {
            "p1": [main_sub_tip, main_sub_eps[0], sub_eps[0], sub_tip],
            "p2": [main_sub_tip, main_sub_eps[1], sub_eps[1], sub_tip],
            "p3": [main_sub_tip, main_sub_eps[0], main_eps1[ind], main_tip1, main_tip2, main_eps2[ind], main_sub_eps[1]],
            "p4": [main_tip1, main_tip2, main_eps1[closest_main_segment], main_eps2[closest_main_segment]]
        })

    def sort_roof_segments(self, roof_type, roof_id):

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

        return self.create_polygons(roof_type, labeled_points)

    def create_polygons(self, roof_type, labeled_points):

        polygon_points = []

        if roof_type=="gabled":
            polygon_points.append(labeled_points["p1"])
            polygon_points.append(labeled_points["p2"])
        elif roof_type=="flat":
            polygon_points.append(labeled_points["p1"])
        elif roof_type=="t-element":
            polygon_points.append(MultiPoint(labeled_points["p1"]).convex_hull)
            polygon_points.append(MultiPoint(labeled_points["p2"]).convex_hull)
            polygon_points.append(Polygon(labeled_points["p3"]))
            polygon_points.append(MultiPoint(labeled_points["p4"]).convex_hull)
            return polygon_points
        else:
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

                match roof_type:
                    case "corner_element":
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
                    case "hipped":
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
        return [MultiPoint(coords).convex_hull for coords in polygon_points]

    def fit(self):

        self.roofs_df = self.fit_planes_to_roof_segments()
        self.intersection_df = self.match_planes()

        new_rows = {
            "roof_type": [],
            "roof_id": [],
            "geometry": []
        }

        roof_types = ['hipped', 'corner_element', 'gabled', 'flat', 't-element']

        for roof_type in roof_types:
            ids = list(set(list(self.roofs_df.loc[self.roofs_df['roof_type'] == roof_type]['roof_id'])))

            for id in ids:
                new_rows["roof_type"].append(roof_type)
                new_rows["roof_id"].append(id)

                match roof_type:
                    case "corner_element":
                        new_rows["geometry"].append(self.sort_roof_segments(roof_type, id))
                    case "hipped":
                        new_rows["geometry"].append(self.sort_roof_segments(roof_type, id))
                    case "gabled":
                        new_rows["geometry"].append(self.sort_gabled_roof_segments(roof_type, id))
                    case "flat":
                        new_rows["geometry"].append(self.sort_flat_roof_segment(roof_type, id))
                    case "t-element":
                        new_rows["geometry"].append(self.sort_telement_roof_segments(roof_type, id))

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
                    meta_roof = self.roofs_df.loc[self.roofs_df['roof_id'] == id]
                    roof = self.polygon_roofs_df.loc[self.polygon_roofs_df['roof_id'] == id]
                    polygons = list(roof['geometry'])[0]
                    x, y, z, s, rgb = meta_roof["x_coordinates"], meta_roof["y_coordinates"], meta_roof["z_coordinates"], 10, 'r'
                    Plotter.double_segmented_polygons_3d(id, polygons, polygons, list(x), list(y), list(z), s, rgb)



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