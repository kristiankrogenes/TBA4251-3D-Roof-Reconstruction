import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from shapely import MultiPolygon, LineString
import os 

class Plotter():

    def scatterplot_3d(roof_id, roof_type, x, y, z, size, color, sources):
        name = "scatterplot_3d"
        fig = plt.figure()
        colors = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
        # poly_colors = [colors[i] for i in sources]
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, s=size/4, c=color)

        if not os.path.exists(f"./data/plots/scatterplots"):
            os.mkdir(f"./data/plots/scatterplots")
        plt.savefig(f"./data/plots/scatterplots/{roof_type}_{roof_id}.png", bbox_inches='tight')

        plt.title(f'{roof_type} - {roof_id}')
        plt.show()

    def polygons_2d(polygon):

        fig, ax = plt.subplots()

        if isinstance(polygon, MultiPolygon):
            for poly in polygon.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, 'b-')
        else:
            x, y = polygon.exterior.xy
            ax.plot(x, y, 'b-')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Shapely Polygon Plot')
        ax.set_aspect('equal')

        plt.show()
    
    def segmented_polygons_3d(roof_id, roof_type, roof):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        xmins, xmaxs = [], []
        ymins, ymaxs = [], []
        zmins, zmaxs = [], []
        for polygon in roof:
            roof_3d = [(x, y, z) for x, y, z in polygon.exterior.coords]

            xmin, xmax = min([coord[0] for coord in roof_3d]), max([coord[0] for coord in roof_3d])
            ymin, ymax = min([coord[1] for coord in roof_3d]), max([coord[1] for coord in roof_3d])
            zmin, zmax = min([coord[2] for coord in roof_3d]), max([coord[2] for coord in roof_3d])

            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            zmins.append(zmin)
            zmaxs.append(zmax)

            poly_collection = Poly3DCollection([roof_3d], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)

            ax.add_collection3d(poly_collection)

        ax.set_title(f"{roof_type} - {roof_id}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([min(xmins) - 1, max(xmaxs) + 1])
        ax.set_ylim([min(ymins) - 1, max(ymaxs) + 1])
        ax.set_zlim([min(zmins) - 1, max(zmaxs) + 1])

        if not os.path.exists(f"./data/plots/roof_polygons"):
            os.mkdir(f"./data/plots/roof_polygons")
        plt.savefig(f"./data/plots/roof_polygons/{roof_type}_{roof_id}.png", bbox_inches='tight')

        plt.show()

    def double_segmented_polygons_3d(id, roof1, roof2, footprint, x, y, z, s, c):
        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        
        # for i in range(len(x)):
        #     ax1.scatter(x[i], y[i], z[i], s=s/4, c=c)

        xmins, xmaxs = [], []
        ymins, ymaxs = [], []
        zmins, zmaxs = [], []
        # Convert Shapely Polygons to 3D polygons
        for p1, p2 in zip(roof1, roof2):
            roof1_3d = [(x, y, z) for x, y, z in p1.exterior.coords]
            roof2_3d = [(x, y, z) for x, y, z in p2.exterior.coords]

            xmin, xmax = min([coord[0] for coord in roof1_3d]), max([coord[0] for coord in roof1_3d])
            ymin, ymax = min([coord[1] for coord in roof1_3d]), max([coord[1] for coord in roof1_3d])
            zmin, zmax = min([coord[2] for coord in roof1_3d]), max([coord[2] for coord in roof1_3d])
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            zmins.append(zmin)
            zmaxs.append(zmax)

            # Create Poly3DCollection for the 3D polygons
            poly1_collection = Poly3DCollection([roof1_3d], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)
            poly2_collection = Poly3DCollection([roof2_3d], facecolors='c', linewidths=1, edgecolors='g', alpha=0.25)

            # Add the Poly3DCollections to the 3D axis
            ax1.add_collection3d(poly1_collection)
            ax2.add_collection3d(poly2_collection)

            ax2.scatter(roof2_3d[0][0], roof2_3d[0][1], roof2_3d[0][2], s=s/4, c='r')

        # fpx, fpy = footprint.exterior.xy
        # fpz = np.array([max(zmaxs) -9 for _ in range(len(fpx))])

        # footprint_3d = [(x, y, max(zmaxs) -9) for x, y in zip(fpx, fpy)]
        # footprint_collection = Poly3DCollection([footprint_3d], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)
        # ax2.add_collection3d(footprint_collection)
        # ax2.scatter(fpx, fpy, fpz, s=s/4, c=['b']+['r' for _ in range(len(fpx)-1)])

        # bottom_points = np.array(footprint)
        bx, by, bz = [x for x, _, _ in footprint],[y for _, y, _ in footprint], [z for _, _, z in footprint]
        ax2.scatter(bx, by, bz, s=s/4, c='r')

        ax1.set_title(f"Roof polygons for hipped roof, ID: {id}")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2.set_title(f"Roof polygons for hipped roof, ID: {id}")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        ax1.set_xlim([min(xmins) - 1, max(xmaxs) + 1])
        ax1.set_ylim([min(ymins) - 1, max(ymaxs) + 1])
        ax1.set_zlim([min(zmins) - 1, max(zmaxs) + 1])
        ax2.set_xlim([min(xmins) - 1, max(xmaxs) + 1])
        ax2.set_ylim([min(ymins) - 1, max(ymaxs) + 1])
        ax2.set_zlim([max(zmaxs) -10, max(zmaxs) + 1])

        plt.show()
    
    def planes_3d(roof_id, roof_type, planes, coords):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red', 'cyan', 'green', 'yellow', 'purple', 'orange', 'blue']

        min_boundaries = [np.inf, np.inf, np.inf]
        max_boundaries = [-np.inf,- np.inf, -np.inf]

        for i, (plane_coeffs, seg_coords) in enumerate(zip(planes, coords)):
            A, B, C, D = plane_coeffs
            x, y, z = seg_coords
            xyz_min = [min(x), min(y), min(z)]
            xyz_max = [max(x), max(y), max(z)]

            min_boundaries = [xyz_min[i] if xyz_min[i] < min_boundaries[i] else min_boundaries[i] for i in range(3)]
            max_boundaries = [xyz_max[i] if xyz_max[i] > max_boundaries[i] else max_boundaries[i] for i in range(3)]

            x_grid = np.linspace(xyz_min[0], xyz_max[0])
            y_grid = np.linspace(xyz_min[1], xyz_max[1])
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = (-A*X-B*Y-D) / C

            ax.plot_surface(X, Y, Z, color=colors[i], alpha=0.4)
            ax.scatter(x, y, z, s=10/4, c=colors[-i])
            # else:
            #     x_grid = np.linspace(min_boundaries[0], max_boundaries[0])
            #     y_grid = np.linspace(min_boundaries[1], max_boundaries[1])
            #     X, Y = np.meshgrid(x_grid, y_grid)
            #     Z = plane_coeffs[0]*X + plane_coeffs[1]*Y + plane_coeffs[2]
            #     ax.plot_surface(X, Y, Z, color=colors[i], alpha=0.4)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([min_boundaries[0]-1, max_boundaries[0]+1])
        ax.set_ylim([min_boundaries[1]-1, max_boundaries[1]+1])
        ax.set_zlim([min_boundaries[2]-1, max_boundaries[2]+1])

        if not os.path.exists(f"./data/plots/lsm_planes"):
            os.mkdir(f"./data/plots/lsm_planes")
        plt.savefig(f"./data/plots/lsm_planes/{roof_type}_{roof_id}.png", bbox_inches='tight')

        plt.show()


    def roof_polygons_with_building_footprint(roof_id, roof_type, roof, footprint):
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection='3d')

        xmins, xmaxs = [], []
        ymins, ymaxs = [], []
        zmins, zmaxs = [], []

        for polygon in roof:
            roof_3d = [(x, y, z) for x, y, z in polygon.exterior.coords]

            xmin, xmax = min([coord[0] for coord in roof_3d]), max([coord[0] for coord in roof_3d])
            ymin, ymax = min([coord[1] for coord in roof_3d]), max([coord[1] for coord in roof_3d])
            zmin, zmax = min([coord[2] for coord in roof_3d]), max([coord[2] for coord in roof_3d])
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            zmins.append(zmin)
            zmaxs.append(zmax)

            poly_collection = Poly3DCollection([roof_3d], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)

            ax.add_collection3d(poly_collection)

        fpx, fpy = footprint.exterior.xy
        footprint_coords = [(x, y, min(zmins)-5) for x, y in zip(fpx, fpy)]
        footprint_collection = Poly3DCollection([footprint_coords], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)
        ax.add_collection3d(footprint_collection)
        
        ax.set_title(f"{roof_type} - {roof_id}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([min(xmins) - 1, max(xmaxs) + 1])
        ax.set_ylim([min(ymins) - 1, max(ymaxs) + 1])
        ax.set_zlim([min(zmins) - 6, max(zmaxs) + 1])

        if not os.path.exists(f"./data/plots/footprints"):
            os.mkdir(f"./data/plots/footprints")
        plt.savefig(f"./data/plots/footprints/{roof_type}_{roof_id}.png", bbox_inches='tight')

        plt.show()

    def lo2d_buildings(roof_id, roof_type, roof_polygons, wall_polygons, pointcloud=False):
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection='3d')

        for roof_polygon in roof_polygons:
            poly_collection = Poly3DCollection([roof_polygon.exterior.coords], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)
            ax.add_collection3d(poly_collection)
        
        for wall_polygon in wall_polygons:
            poly_collection = Poly3DCollection([wall_polygon.exterior.coords], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)
            ax.add_collection3d(poly_collection)
        
        if pointcloud:
            x, y, z = pointcloud
            for i in range(len(x)):
                ax.scatter(x[i], y[i], z[i], s=10/4, c='r')
        
        x_min, y_min, z_min = float('inf'), float('inf'), float('inf')
        x_max, y_max, z_max = float('-inf'), float('-inf'), float('-inf')

        for polygon in roof_polygons + wall_polygons:
            for x, y, z in polygon.exterior.coords:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                z_min = min(z_min, z)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                z_max = max(z_max, z)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        

        ax.set_title(f"Lo2D: {roof_type} / {roof_id}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if pointcloud:
            if not os.path.exists(f"./data/plots/lo2d_models_with_pc"):
                os.mkdir(f"./data/plots/lo2d_models_with_pc")
            plt.savefig(f"./data/plots/lo2d_models_with_pc/{roof_type}_{roof_id}.png", bbox_inches='tight')
        else:
            if not os.path.exists(f"./data/plots/lo2d_models_with_pc"):
                os.mkdir(f"./data/plots/lo2d_models_with_pc")
            plt.savefig(f"./data/plots/lo2d_models_with_pc/{roof_type}_{roof_id}.png", bbox_inches='tight')

        plt.show()