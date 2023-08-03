import matplotlib.pyplot as plt

class plotLineGraphFromDictionary:
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
    def __init__(self, data, end=100, start=0, title='', xlabel='', ylabel=''):

        """
        draw line graph. Plot the input data key as x and value as y
        :param end: end of graph to plot
        :type end: int
        :param start: start fo graph to plot
        :type start: int
        :param title: title of graph
        :type title: str
        :param xlabel: xlabel of graph
        :type xlabel: str
        :param ylabel: ylabel of grapth
        :type ylabel: str
        """
        end = int(len(data) * end / 100)
        start = int(len(data) * start / 100)
        x = tuple(data.keys())[start:end]
        y = tuple(data.values())[start:end]
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)