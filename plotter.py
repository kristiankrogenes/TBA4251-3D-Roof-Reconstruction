import matplotlib.pyplot as plt

class Plotter():

    def scatterplot_3d(x, y, z, color, size):
        name = "scatterplot_3d"
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, c=color, s= size/4)
        # if not os.path.exists(f"../images/{name}"):
        #     os.mkdir(f"../images/{name}")
        # plt.savefig(f"../images/{name}/{roof}.png", bbox_inches='tight')
        plt.show()