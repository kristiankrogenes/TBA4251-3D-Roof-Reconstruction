from plotter import Plotter

from models.gabled import GabledRoof
from models.flat import FlatRoof
from models.hipped import HippedRoof

from helpers import extract_variables_from_las_object, is_equal_color
import os
import laspy
import numpy as np


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

    def fit(self):

        roof_types = ['hipped', 'corner_element', 'gabled', 'flat', 't-element', 'cross_element']

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
    
    def plot_data(self, plot_type):
        match plot_type:
            case "pointcloud_scatterplot_3d":
                for roof_type in self.roof_data.roofs:
                    for roof_id in self.roof_data.roofs[roof_type]:
                        plot_title = f'{roof_type}-{roof_id}'
                        Plotter.scatterplot_3d(plot_title, *extract_variables_from_las_object(self.roof_data.roofs[roof_type][roof_id]))
            case "roof_polygons_3d_v2":
                for roof in self.roof_objects:
                    Plotter.double_segmented_polygons_3d(
                        roof.roof_id, 
                        roof.roof_polygons, 
                        roof.roof_polygons, 
                        roof.segment_xs, 
                        roof.segment_ys,
                        roof.segment_zs, 
                        10, 
                        'r'
                    )

if __name__ == "__main__":

    DATA_FOLDER_PATH = './data/roofs'

    lo2d = Lo2D(DATA_FOLDER_PATH)
    lo2d.fit()

    # lo2d.plot_data("pointcloud_scatterplot_3d")
    # lo2d.plot_data("planes_3d")
    # lo2d.plot_data("roof_polygons_3d")
    lo2d.plot_data("roof_polygons_3d_v2")