import matplotlib.pyplot as plt

class plotLineGraph:
    """
    This class plot graph of input data

    Attributes:
    ----------
    data : dict
        store input data as dict

    Methods:
    -------
    plotLineGraph()
        draw line graph of input data. input data's key is x and value is y.
    """
    def __init__(self, data):
        self.data = data

    def plotLineGraph(self, percentage=100, title='', xlabel='', ylabel=''):
        """
        draw line graph. Plot the input data key as x and value as y
        :param percentage: percentage of graph to plot
        :type percentage: int
        :param title: title of graph
        :type title: str
        :param xlabel: xlabel of graph
        :type xlabel: str
        :param ylabel: ylabel of grapth
        :type ylabel: str
        """
        numberOfGraphToPlot = int(len(self.data) * percentage / 100)
        x = tuple(self.data.keys()[:numberOfGraphToPlot])
        y = tuple(self.data.values()[:numberOfGraphToPlot])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(x, y)
        #plt.plot(x, y)
        plt.show()