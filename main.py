from plotter import Plotter
from helpers import fetch_roof_data, extract_variables_from_las_segments


if __name__ == "__main__":

    roofs = fetch_roof_data()

    # Visualize roof data with scatter plot
    for roof in roofs:
        roof_segments = [roof_segment for roof_segment in roofs[roof].values()]
        Plotter.scatterplot_3d(*extract_variables_from_las_segments(roof_segments))