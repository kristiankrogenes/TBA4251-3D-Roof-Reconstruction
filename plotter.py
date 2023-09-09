import matplotlib.pyplot as plt
import geopandas as gpd

class Plotter():

    def scatterplot_3d(x, y, z, color, size):
        name = "scatterplot_3d"
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, c=color, s=size/4)
        # if not os.path.exists(f"../images/{name}"):
        #     os.mkdir(f"../images/{name}")
        # plt.savefig(f"../images/{name}/{roof}.png", bbox_inches='tight')
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