from plotter import Plotter
from helpers import fetch_roof_data, extract_variables_from_las_segments, create_polygon_from_roof_segment, create_roof_polygons
import os
import laspy
import numpy as np
import alphashape 
import geopandas as gpd
import pandas as pd

class Lo2D():

    def __init__(self, folder_path):
        self.roofs = self.fetch_data_from_point_cloud(folder_path)

    def fetch_data_from_point_cloud(self, folder_path):
        DATA_FOLDERS = os.listdir(folder_path)

        data = {}

        for data_folder in DATA_FOLDERS:
            if data_folder != '.DS_Store':
                data[data_folder] = {}
                FILES_PATH = f'{folder_path}/{data_folder}/'
                file_names = os.listdir(FILES_PATH)

                for file_name in file_names:
                    FILE_PATH = f'{FILES_PATH}/{file_name}'
                    las = laspy.read(FILE_PATH)
                    data[data_folder][file_name[:-4]] = las
        return data 
    
    def get_alpha_shape_polygons(self, roof_segments):
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

    def create_roofs_df(self):

        df_roof_polygons = gpd.GeoDataFrame()

        for roof in self.roofs:
            roof_segments = [roof_segment for roof_segment in self.roofs[roof].values()]
            df = self.get_alpha_shape_polygons(roof_segments)
            df["roof_id"] = roof
            z_coords = [point[2] for point in list(df.iloc[0].geometry.exterior.coords)]
            df["minimum_z"] = min(z_coords)
            df_roof_polygons = pd.concat([df_roof_polygons, df])

            df_roof_polygons = df_roof_polygons.reset_index(drop=True)
        self.df_roof_polygons = df_roof_polygons
    
    def plot_data(self, plot_type):
        match plot_type:
            case "pointcloud_scatterplot_3d":
                for roof in self.roofs:
                    roof_segments = [roof_segment for roof_segment in self.roofs[roof].values()]
                    Plotter.scatterplot_3d(*extract_variables_from_las_segments(roof_segments))
            case "segmented_roof_polygons_2d":
                roof_ids = list(set(list(self.df_roof_polygons['roof_id'])))
                for id in roof_ids:
                    roof_segments = self.df_roof_polygons.loc[self.df_roof_polygons['roof_id'] == id]
                    polygons = [poly for poly in list(roof_segments['geometry'])]
                    Plotter.segmented_polygons_2d(polygons)



if __name__ == "__main__":

    DATA_FOLDER_PATH = './roofs'

    lo2d = Lo2D(DATA_FOLDER_PATH)

    lo2d.create_roofs_df()
    lo2d.plot_data("pointcloud_scatterplot_3d")


    
