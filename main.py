from plotter import Plotter

from models.gabled import GabledRoof
from models.flat import FlatRoof
from models.hipped import HippedRoof

from helpers import extract_variables_from_las_object, is_equal_color, get_upper_points, create_concave_polygon, order_points_clockwise, find_min_max_values, order_points_left_to_right, distance
import os
import laspy
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from shapely import MultiPolygon, MultiPoint, Polygon
from scipy.spatial import Delaunay
from alphashape import alphashape


class RoofLasData():

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

class Lo2D():

    def __init__(self, folder_path):
        self.roof_data = RoofLasData(folder_path)

    def extract_roof_data(self, roof_type, roof_id):
        segment_ids = list(set(self.roof_data.roofs[roof_type][roof_id].point_source_id))
        segment_xs, segment_ys, segment_zs = [], [], []
        for sid in segment_ids:
            roof_segment_indexes = np.array([id==sid for id in self.roof_data.roofs[roof_type][roof_id].point_source_id])
            roof_segment_coords = np.array(self.roof_data.roofs[roof_type][roof_id].xyz)[roof_segment_indexes]

            x, y, z = roof_segment_coords[:, 0], roof_segment_coords[:, 1], roof_segment_coords[:, 2]
            segment_xs.append(x)
            segment_ys.append(y)
            segment_zs.append(z)
        return roof_type, roof_id, segment_ids, segment_xs, segment_ys, segment_zs

    def create_polygon_roofs(self):
        roof_types = ['hipped', 'corner_element', 'gabled', 'flat', 't-element', 'cross_element']
        # roof_types = ['gabled']

        roof_objects = []
        for roof_type in roof_types:

            roof_ids = list(self.roof_data.roofs[roof_type].keys())

            for roof_id in roof_ids:
                match roof_type:
                    case "corner_element":
                        corner_element_roof = HippedRoof(*self.extract_roof_data(roof_type, roof_id))
                        roof_objects.append(corner_element_roof)
                    case "hipped":
                        hipped_roof = HippedRoof(*self.extract_roof_data(roof_type, roof_id))
                        roof_objects.append(hipped_roof)
                    case "gabled":
                        gabled_roof = GabledRoof(*self.extract_roof_data(roof_type, roof_id))
                        roof_objects.append(gabled_roof)
                    case "flat":
                        flat_roof = FlatRoof(*self.extract_roof_data(roof_type, roof_id))
                        roof_objects.append(flat_roof)
                    case "t-element":
                        t_element_roof = GabledRoof(*self.extract_roof_data(roof_type, roof_id))
                        roof_objects.append(t_element_roof)
                    case "cross_element":
                        cross_element_roof = GabledRoof(*self.extract_roof_data(roof_type, roof_id))
                        roof_objects.append(cross_element_roof)

        self.roof_objects = roof_objects

    def create_dataframe(self):
        df = []
        for roof in self.roof_objects:
            for i, segment in enumerate(roof.roof_polygons):
                df.append({
                    "roof_id": roof.roof_id,
                    "roof_type": roof.roof_type,
                    "geometry": segment,
                    "classification": i,
                })
        self.roof_df = gpd.GeoDataFrame(df)

    def merge_buildings(self, buildings, roofs_with_building, roofs_with_several_buildings):
        """
        Merge buildings when a roof has several buildings
        """
        roofs_several_buildings = roofs_with_building.query("roof_id in @roofs_with_several_buildings").sort_values(by=["roof_id", "classification", "index_right"])
        for roof in roofs_several_buildings.roof_id.unique():
            _roof = roofs_with_building.query("roof_id == @roof")
            building_indices = _roof.index_right.unique()
            _buildings = buildings.iloc[building_indices]
            merged = unary_union(_buildings.geometry.to_list())
            for b in _buildings.index:
                buildings.at[b, 'geometry'] = merged
        return buildings

    def find_building_footprints(self):

        trondheim_buildings = gpd.read_file("./buildings_trondheim")

        lo2d.roof_df.crs = 25832
        new_df = lo2d.roof_df.to_crs(trondheim_buildings.crs)

        matched_roof_segments_with_buildings = gpd.sjoin_nearest(new_df, trondheim_buildings)

        roofs_with_several_buildings = matched_roof_segments_with_buildings[["roof_id", "index_right"]].groupby("roof_id").describe().index_right.query("std > 0").index
        merged_buildings = self.merge_buildings(trondheim_buildings, matched_roof_segments_with_buildings, roofs_with_several_buildings)

        relevant_building_ids = matched_roof_segments_with_buildings.index_right.unique()
        relevant_buildings = merged_buildings.iloc[relevant_building_ids]

        matched_roofs_with_buildings = matched_roof_segments_with_buildings.merge(relevant_buildings, on="osm_id")
        matched_roofs_with_buildings = matched_roofs_with_buildings.drop(["geometry_x", "classification", "code_x", "fclass_x", "name_x", "type_x", "code_y", "fclass_y", "name_y", "type_y"], axis=1)
        matched_roofs_with_buildings = matched_roofs_with_buildings.drop_duplicates(subset='roof_id', keep='first')

        matched_roofs_with_buildings = gpd.GeoDataFrame(matched_roofs_with_buildings, geometry=list(matched_roofs_with_buildings["geometry_y"]))
        matched_roofs_with_buildings.crs = 4326
        matched_roofs_with_buildings = matched_roofs_with_buildings.to_crs(epsg=25832)

        for i in range(len(matched_roofs_with_buildings)):
            for roof in self.roof_objects:
                if roof.roof_id == matched_roofs_with_buildings.iloc[i]["roof_id"]:
                    roof.footprint = matched_roofs_with_buildings.iloc[i]["geometry"]

    def match_footprints_with_roof_polygons(self):
        for roof in self.roof_objects:
            print("Roof-------", roof.roof_id)

            upper_footprint = []

            updated_segment_points = {}
            footprint_corner_points_used = {}
            updated_bottom_points = []

            if isinstance(roof.footprint, MultiPolygon):
                poly_area = 0
                index = None
                for i, poly in enumerate(roof.footprint.geoms):
                    if poly.area > poly_area:
                        poly_area = poly_area
                        index = i
                roof.footprint = roof.footprint.geoms[index]

            for roof_plane, roof_polygon, segment_id in zip(roof.segment_planes, roof.roof_polygons, roof.segment_ids):
                if roof.roof_type == "flat":
                    upper_points = []
                else:
                    upper_points = get_upper_points(roof_polygon)

                is_main_gabled = False
                if roof.roof_type == "cross_element":
                    is_main_gabled = segment_id in roof.main_gabled_segments

                updated_polygon_points = [[p[0], p[1], p[2]] for p in upper_points]
                updated_segment_points[segment_id] = {}
                updated_segment_points[segment_id]["upper_points"] = upper_points
                footprint_corner_points_used[segment_id] = [False for _ in range(len(roof.footprint.exterior.coords[:-1]))]

                corner_point_distances = []
                # corner_point_indexes = []
                for i, corner_point in enumerate(roof.footprint.exterior.coords[:-1]):
                    x, y = corner_point
                    A, B, C, D = roof_plane
                    z = (-A*x -B*y -D) / C
                    if z > roof.global_minz and z < roof.global_maxz:
                        corner_point_distances.append(sum([distance(corner_point, rp_point) for rp_point in roof_polygon.exterior.coords]))
                        # corner_point_indexes.append(i)
                        updated_polygon_points.append([x, y, z])
                        footprint_corner_points_used[segment_id][i] = True

                if not is_main_gabled and roof.roof_type == "cross_element":
                    incorrect_corner_points = sorted(corner_point_distances)[2:]
                    indexes = [corner_point_distances.index(dist) for dist in incorrect_corner_points] 
                    points_to_remove = [updated_polygon_points[len(upper_points)+j] for j in indexes]
                    indexes_to_remove = [j for j in range(len(corner_point_distances)) if corner_point_distances[j] in incorrect_corner_points]
                    INDEXES_to_remove = [j for j in range(len(footprint_corner_points_used[segment_id])) if footprint_corner_points_used[segment_id][j]]
                    for point_to_remove, index_to_remove in zip(points_to_remove, indexes_to_remove):
                        updated_polygon_points.remove(point_to_remove)
                        footprint_corner_points_used[segment_id][INDEXES_to_remove[index_to_remove]] = False

                updated_segment_points[segment_id]["bottom_points"] = np.array(updated_polygon_points)[len(upper_points):]

            updated_corners = []
            for sid in updated_segment_points.keys():

                upper_points = updated_segment_points[sid]["upper_points"]
                bottom_polygon_points = list(updated_segment_points[sid]["bottom_points"])

                updated_polygon_points = upper_points + bottom_polygon_points

                count1 = -1
                for i, is_corner_used in enumerate(footprint_corner_points_used[sid]):
                    if is_corner_used and i not in updated_corners:
                        count1 += 1
                        for sid2 in footprint_corner_points_used.keys():
                            if sid != sid2:
                                if footprint_corner_points_used[sid2][i]:
                                    
                                    count2 = len([1 for i in range(i) if footprint_corner_points_used[sid2][i]])

                                    p1 = updated_segment_points[sid]["bottom_points"][count1]
                                    p2 = updated_segment_points[sid2]["bottom_points"][count2]

                                    avg_point = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]
                                    updated_segment_points[sid]["bottom_points"][count1] = avg_point
                                    updated_segment_points[sid2]["bottom_points"][count2] = avg_point

                if len(updated_polygon_points) <= 5:
                    upper_footprint.append(MultiPoint(updated_polygon_points).convex_hull)
                else:
                    if roof.roof_type == "flat":
                        upper_footprint.append(Polygon(updated_polygon_points))
                    elif roof.roof_type == "t-element" or roof.roof_type == "cross_element":
                        x = [x for x, _, _ in upper_points]
                        y = [y for _, y, _ in upper_points]
                        z = [z for _, _, z in upper_points]
                        minxp, maxxp, _, _, _, _ = find_min_max_values(x, y, z)
                        point_to_remove = [p for p in upper_points if p not in [minxp, maxxp]][0]
                        upper_points.remove(point_to_remove)

                        updated_polygon_points = np.array(updated_polygon_points)
                        bottom_polygon_points = updated_polygon_points[len(upper_points)+1:]
                        ordered_bottom_points = order_points_left_to_right(bottom_polygon_points)
                        ordered_bottom_points.insert(2, point_to_remove)

                        ordered_upper_points = order_points_left_to_right(upper_points)
                        ordered_points = ordered_bottom_points + list(reversed(ordered_upper_points))
                        upper_footprint.append(Polygon(ordered_points)) 
                    else:  
                        updated_polygon_points = np.array(updated_polygon_points)
                        bottom_polygon_points = updated_polygon_points[len(upper_points):]
                        ordered_bottom_points = order_points_clockwise(bottom_polygon_points)
                        ordered_upper_points = order_points_clockwise(upper_points)

                        ordered_points = ordered_bottom_points + list(reversed(ordered_upper_points))
                        upper_footprint.append(Polygon(ordered_points)) 

            for sid in updated_segment_points.keys():
                for p in list(updated_segment_points[sid]["bottom_points"]):
                    already_exists = False
                    for added_point in updated_bottom_points:
                        if distance(added_point, p) == 0:
                            already_exists = True
                            break
                    if not already_exists:
                        updated_bottom_points.append(list(p))

            roof.upper_footprint_bottom_points = updated_bottom_points

            roof.upper_footprint = upper_footprint

    def create_building_walls(self):
        for roof in self.roof_objects:
            print("Roof ==========", roof.roof_id)

            bottom_points = roof.upper_footprint_bottom_points
            footprint_points = roof.footprint.exterior.coords[:-1]
            footprint_points = [[x, y, roof.global_minz-5] for x, y in footprint_points]
            print(footprint_points)
            wall_polygons = []

            for i, corner_point in enumerate(footprint_points):
                corner_to_bottom_distances1 = []
                corner_to_bottom_distances2 = []
                for bottom_point in bottom_points:
                    p1 = np.array(footprint_points[i])[:2]
                    p2 = np.array(bottom_point)[:2]
                    corner_to_bottom_distances1.append(distance(p1, p2))
                    if i+1==len(footprint_points):
                        p3 = np.array(footprint_points[0])[:2]
                        corner_to_bottom_distances2.append(distance(p3, p2))
                    else:
                        p3 = np.array(footprint_points[i+1])[:2]
                        corner_to_bottom_distances2.append(distance(p3, p2))

                cbp1 = bottom_points[corner_to_bottom_distances1.index(min(corner_to_bottom_distances1))]
                cbp2 = bottom_points[corner_to_bottom_distances2.index(min(corner_to_bottom_distances2))]
                
                if i+1==len(footprint_points):
                    wall_polygons.append(Polygon([cbp1, footprint_points[i], footprint_points[0], cbp2]))
                else:
                    wall_polygons.append(Polygon([cbp1, footprint_points[i], footprint_points[i+1], cbp2]))

            roof.wall_polygons = wall_polygons
                

    def fit(self):
        self.create_polygon_roofs()
        self.create_dataframe()
        self.find_building_footprints()
        self.match_footprints_with_roof_polygons()
        self.create_building_walls()

    def plot_data(self, plot_type):
        match plot_type:
            case "pointcloud_scatterplot_3d":
                for roof_type in self.roof_data.roofs:
                    for roof_id in self.roof_data.roofs[roof_type]:
                        # plot_title = f'{roof_type}-{roof_id}'
                        Plotter.scatterplot_3d(roof_id, roof_type, *extract_variables_from_las_object(self.roof_data.roofs[roof_type][roof_id]))
            case "roof_segment_planes_3d":
                for roof in self.roof_objects:
                    # plot_title = f'{roof_type}-{roof_id}'
                    coords = [[x, y, z] for x, y, z in zip(roof.segment_xs, roof.segment_ys, roof.segment_zs)]
                    Plotter.planes_3d(roof.roof_id, roof.roof_type, roof.segment_planes, coords)
            case "roof_polygons_3d":
                for roof in self.roof_objects:
                    Plotter.segmented_polygons_3d(
                        roof.roof_id,
                        roof.roof_type,
                        roof.roof_polygons,
                        # roof.upper_footprint,
                    )
            case "roof_polygons_3d_v2":
                for roof in self.roof_objects:
                    Plotter.double_segmented_polygons_3d(
                        roof.roof_id,
                        roof.roof_polygons,
                        roof.upper_footprint,
                        roof.upper_footprint_bottom_points,
                        roof.segment_xs,
                        roof.segment_ys,
                        roof.segment_zs,
                        10,
                        'r'
                    )
            case "roof_polygons_with_footprint":
                for roof in self.roof_objects:
                    Plotter.roof_polygons_with_building_footprint(
                        roof.roof_id,
                        roof.roof_type,
                        roof.roof_polygons,
                        roof.footprint,
                    )
            case "lo2d_buildings":
                for roof in self.roof_objects:
                    Plotter.lo2d_buildings(
                        roof.roof_id,
                        roof.roof_type,
                        roof.upper_footprint,
                        roof.wall_polygons,
                        [roof.segment_xs, roof.segment_ys, roof.segment_zs],
                    )

if __name__ == "__main__":

    DATA_FOLDER_PATH = './data/roofs'

    lo2d = Lo2D(DATA_FOLDER_PATH)
    lo2d.fit()

    # lo2d.plot_data("pointcloud_scatterplot_3d")
    # lo2d.plot_data("roof_segment_planes_3d")
    # lo2d.plot_data("roof_polygons_3d")
    # lo2d.plot_data("roof_polygons_3d_v2")
    # lo2d.plot_data("roof_polygons_with_footprint")
    lo2d.plot_data("lo2d_buildings")