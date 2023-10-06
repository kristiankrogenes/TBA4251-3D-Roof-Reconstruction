import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

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

    def segmented_polygons_2d(id, polygons):
        name = "segmented_polygons_2d"
        colors = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
        poly_colors = [colors[i] for i in range(len(polygons))]
        myPoly = gpd.GeoSeries(polygons)
        myPoly.plot(color=poly_colors)
        plt.title(id)
        plt.show()

    def double_segmented_polygons_2d(id, roof1, roof2):
        name = "segmented_polygons_2d"
        colors = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']

        poly_colors1 = [colors[i] for i in range(len(roof1))]
        myPoly1 = gpd.GeoSeries(roof1)
        # myPoly1.plot(color=poly_colors1)

        poly_colors2 = [colors[i] for i in range(len(roof2))]
        myPoly2 = gpd.GeoSeries(roof2)
        # myPoly2.plot(color=poly_colors2)

        # plt.title(id)
        # plt.show()

        # fig, ax = plt.subplots()
        # myPoly1.plot(color=poly_colors1, ax=ax)
        # myPoly2.plot(color=poly_colors2, ax=ax)
        # plt.title(id)
        # plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with two subplots
        myPoly1.plot(color=poly_colors1, ax=axes[0])  # Plot in the first subplot
        myPoly2.plot(color=poly_colors2, ax=axes[1])  # Plot in the second subplot

        axes[0].set_title("Plot 1")  # Set title for the first subplot
        axes[1].set_title("Plot 2")  # Set title for the second subplot

        plt.suptitle(id)  # Set a common title for the entire figure
        plt.show()

    def double_segmented_polygons_3d(id, roof1, roof2, x, y, z, s, c):
        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        for i in range(len(x)):
            ax1.scatter(x[i], y[i], z[i], s=s/4, c=c)

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
        ax2.set_zlim([min(zmins) - 1, max(zmaxs) + 1])

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

    def intersection_lines_3d(points, xmin, xmax, ymin, ymax, zmin, zmax, ips):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        for point in points:
            # print(point[0][0], point[0][1], point[0][2],)
            
            ax.plot(
                [point[0][0], point[1][0]],
                [point[0][1], point[1][1]],
                [point[0][2], point[1][2]],
                color='b', 
                label='Line'
            )

        # points = np.array(points)

        # x1, y1, z1 = points[:,0,0], points[:,0,1], points[:,0,2]
        # x2, y2, z2 = points[:,1,0], points[:,1,1], points[:,1,2]
        ips = np.array(ips)
        x, y, z = ips[:,0], ips[:,1], ips[:,2]
        ax.scatter(x, y, z, s=10/4, c='r')
        # ax.scatter(x2, y2, z2, s=10/4, c='b')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([xmin-10, xmax+10])
        ax.set_ylim([ymin-10, ymax+10])
        ax.set_zlim([zmin-10, zmax+10])

        plt.show()
