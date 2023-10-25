import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from shapely import MultiPolygon, LineString

class Plotter():

    def scatterplot_3d(title, x, y, z, size, color, sources):
        name = "scatterplot_3d"
        fig = plt.figure()
        colors = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
        # poly_colors = [colors[i] for i in sources]
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, s=size/4, c=color)
        # if not os.path.exists(f"../images/{name}"):
        #     os.mkdir(f"../images/{name}")
        # plt.savefig(f"../images/{name}/{roof}.png", bbox_inches='tight')
        plt.title(title)
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
    
    def planes_3d(id, planes, coords):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red', 'cyan', 'green', 'yellow', 'purple', 'orange', 'blue']

        min_boundaries = [np.inf, np.inf, np.inf]
        max_boundaries = [-np.inf,- np.inf, -np.inf]

        for i, (plane_coeffs, seg_coords) in enumerate(zip(planes, coords)):
            x, y, z = seg_coords
            xyz_min = [min(x), min(y), min(z)]
            xyz_max = [max(x), max(y), max(z)]

            min_boundaries = [xyz_min[i] if xyz_min[i] < min_boundaries[i] else min_boundaries[i] for i in range(3)]
            max_boundaries = [xyz_max[i] if xyz_max[i] > max_boundaries[i] else max_boundaries[i] for i in range(3)]

            x_grid = np.linspace(xyz_min[0], xyz_max[0])
            y_grid = np.linspace(xyz_min[1], xyz_max[1])
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = plane_coeffs[0]*X + plane_coeffs[1]*Y + plane_coeffs[2]

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

        if True:
            A1, B1, C1 = planes[0]
            A2, B2, C2 = planes[1]
            print("PLANE 1:", A1, B1, C1)
            print("PLANE 2:", A2, B2, C2)

            A = np.array([[A1, B1], [A2, B2]])
            b = np.array([-C1, -C2])

            intersection_point = np.dot(np.linalg.inv(A), b) # Where z=0
            print("INTERSECTION POINT:", intersection_point)

            direction_vector = np.cross([A1, B1, -1], [A2, B2, -1])
            print("DIRECTION VECTOR:", direction_vector)

            zmin, zmax = min(coords[0][2]), max(coords[0][2])
            
            t0 = (zmin)/direction_vector[2]
            t1 = (zmax)/direction_vector[2]

            ax.plot(
                [intersection_point[0]+t0*direction_vector[0], intersection_point[0]+t1*direction_vector[0]], 
                [intersection_point[1]+t0*direction_vector[1], intersection_point[1]+t1*direction_vector[1]], 
                [t0*direction_vector[2], t1*direction_vector[2]],
                color='b', 
                label='Line'
            )

        plt.show()


    def roof_polygons_with_building_footprint(id, roof, footprint, x, y, z, s, c):
        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(121, projection='3d')

        xmins, xmaxs = [], []
        ymins, ymaxs = [], []
        zmins, zmaxs = [], []
        # Convert Shapely Polygons to 3D polygons
        for p1 in roof:
            roof1_3d = [(x, y, z) for x, y, z in p1.exterior.coords]

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

            # Add the Poly3DCollections to the 3D axis
            ax1.add_collection3d(poly1_collection)

        """
        if isinstance(footprint, MultiPolygon):
            for poly in footprint.geoms:
                x, y = poly.exterior.xy
                z = [min(zmins)-5 for _ in range(len(x))]
                # ax1.plot(x, y, 'b-')
                # x, y, z = poly.exterior.coords
                fp = [(xi, yi, zi) for xi, yi, zi in zip(x, y, z)]
                # poly2_collection = Poly3DCollection([poly.exterior.coords], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)
                poly2_collection = Poly3DCollection([fp], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)
                ax1.add_collection3d(poly2_collection)
        else:
            x, y = footprint.exterior.xy
            z = [min(zmins)-5 for _ in range(len(x))]
            # print(list(footprint.exterior.coords))
            # x, y, z = [footprint.exterior.coords
            fp = [(xi, yi, zi) for xi, yi, zi in zip(x, y, z)]
            # poly2_collection = Poly3DCollection([footprint.exterior.coords], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)
            poly2_collection = Poly3DCollection([fp], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)
            ax1.add_collection3d(poly2_collection)
            # ax1.plot(x, y, 'b-')
        """
        for p2 in footprint:
            if isinstance(p2, LineString):
                x, y, z = [list(coord) for coord in zip(*p2.coords)]
                ax1.plot(x, y, z, 'r-')
            else:
                poly2_collection = Poly3DCollection([p2.exterior.coords], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)
                ax1.add_collection3d(poly2_collection)
        

        ax1.set_title(f"Roof polygons for roof, ID: {id}")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax1.set_xlim([min(xmins) - 1, max(xmaxs) + 1])
        ax1.set_ylim([min(ymins) - 1, max(ymaxs) + 1])
        ax1.set_zlim([min(zmins) - 6, max(zmaxs) + 1])

        plt.show()

    def lo2d_buildings(roof_id, roof_type, roof_polygons, wall_polygons):
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection='3d')

        for roof_polygon in roof_polygons:
            poly_collection = Poly3DCollection([roof_polygon.exterior.coords], facecolors='r', linewidths=1, edgecolors='g', alpha=0.25)
            ax.add_collection3d(poly_collection)
        
        for wall_polygon in wall_polygons:
            poly_collection = Poly3DCollection([wall_polygon.exterior.coords], facecolors='b', linewidths=1, edgecolors='g', alpha=0.25)
            ax.add_collection3d(poly_collection)
        
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

        plt.show()