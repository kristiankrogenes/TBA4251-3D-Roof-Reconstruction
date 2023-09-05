import os
import laspy
import numpy as np
import alphashape 
import geopandas as gpd
import pandas as pd

def fetch_roof_data():
    ROOFS_PATH = './roofs'
    ROOF_FOLDERS = os.listdir(ROOFS_PATH)

    roofs = {}

    for roof_folder in ROOF_FOLDERS:
        if roof_folder != '.DS_Store':
            roofs[roof_folder] = {}
            FILES_PATH = f'{ROOFS_PATH}/{roof_folder}/'
            file_names = os.listdir(FILES_PATH)

            for file_name in file_names:
                FILE_PATH = f'{FILES_PATH}/{file_name}'
                las = laspy.read(FILE_PATH)
                roofs[roof_folder][file_name[:-4]] = las
    return roofs

def extract_variables_from_las_segments(las):
    x = list(np.concatenate([list(pc.X) for pc in las]).flat)
    y = list(np.concatenate([list(pc.Y) for pc in las]).flat)
    z = list(np.concatenate([list(pc.Z) for pc in las]).flat)
    c = list(np.concatenate([[las.index(pc) for i in list(pc.X)] for pc in las]).flat)

    # Used to adjust the size of the scatterplot points
    ys = np.random.randint(130, 195, len(x))
    zs = np.random.randint(30, 160, len(y))
    s = zs / ((ys * 0.01) ** 2)

    return x, y, z, c, s

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

def create_roof_polygons(roofs):
    all_roofs = gpd.GeoDataFrame()

    # Find minimum z value for each roof
    for roof in roofs:
        roof_segments = [roof_segment for roof_segment in roofs[roof].values()]
        df = get_alpha_shape_polygons(roof_segments)
        df["roof_id"] = roof
        z_coords = [point[2] for point in list(df.iloc[0].geometry.exterior.coords)]
        df["minimum_z"] = min(z_coords)
        all_roofs = pd.concat([all_roofs, df])

        all_roofs = all_roofs.reset_index(drop=True)
    return all_roofs