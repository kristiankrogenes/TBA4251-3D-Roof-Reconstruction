import os
import laspy
import numpy as np

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