# plotLineGraphFromDictionary is used to convert the given dictionary into plotLineGraph.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.graph import plotLineGraphFromDictionary as plt
#
#     obj = plt.plotLineGraphFromDictionary(idict, 100, 0, " ")
#
#     obj.save()
#
#

__copyright__ = """
 Copyright (C)  2021 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import matplotlib.pyplot as plt

class plotLineGraphFromDictionary:
    """
    This class plot graph of input data

    Attributes:
    ----------
    :param data : dict
        store input data as dict

    Methods:
    -------
    plotLineGraph()
        draw line graph of input data. input data's key is x and value is y.

        **Importing this algorithm into a python program**
        --------------------------------------------------------
        .. code-block:: python

        from PAMI.extras.graph import plotLineGraphFromDictionary as plt

        obj = plt.plotLineGraphFromDictionary(idict, 100, 0, " ")

        obj.save()
    """
    def __init__(self, data: dict, end: int=100, start: int=0, title: str='', xlabel: str='', ylabel: str='') -> None:

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
        x = list(range(len(data)))
        y = list(tuple(data.values())[start:end])
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='.')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

if __name__ == '__main__':
    obj = plotLineGraphFromDictionary(sys.argv[1])
    obj.plotLineGraphFromDictionary(sys.argv[2])