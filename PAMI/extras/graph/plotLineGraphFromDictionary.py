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
import sys
import numbers
import matplotlib.pyplot as plt


class plotLineGraphFromDictionary:
    """
    Plot a dictionary as a line graph.

    The x-axis uses the dictionary keys when they are numbers (for example,
    the output of ``getTransanctionalLengthDistribution()``); the points are
    sorted by key. When the keys are not numbers (for example, item labels from
    ``getSortedListOfItemFrequencies()``), the dictionary order is kept and the
    x-axis shows each entry's rank, starting at 1.

    ``start`` and ``end`` select a percentage range of the entries
    (0 <= start < end <= 100).

    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

            from PAMI.extras.graph import plotLineGraphFromDictionary as plt

            obj = plt.plotLineGraphFromDictionary(idict, 100, 0, "title")

            obj.save("graph.png")
    """

    def __init__(self, data: dict, end: int = 100, start: int = 0, title: str = '', xlabel: str = '',
                 ylabel: str = '') -> None:
        """
        :param data: dictionary to plot; keys are x, values are y
        :type data: dict
        :param end: end of the plotted range, as a percentage of the entries
        :type end: int
        :param start: start of the plotted range, as a percentage of the entries
        :type start: int
        :param title: title of graph
        :type title: str
        :param xlabel: xlabel of graph
        :type xlabel: str
        :param ylabel: ylabel of graph
        :type ylabel: str
        :return: None
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict, got " + type(data).__name__)
        if not 0 <= start < end <= 100:
            raise ValueError("expected 0 <= start < end <= 100, got start=%r, end=%r" % (start, end))

        items = list(data.items())
        numericKeys = len(items) > 0 and all(
            isinstance(k, numbers.Real) and not isinstance(k, bool) for k, _ in items)
        if numericKeys:
            items.sort(key=lambda kv: kv[0])
            positions = [k for k, _ in items]
        else:
            positions = list(range(1, len(items) + 1))

        first = int(len(items) * start / 100)
        last = int(len(items) * end / 100)
        self.x = positions[first:last]
        self.y = [v for _, v in items[first:last]]

        self.figure, self.axes = plt.subplots()
        self.axes.plot(self.x, self.y, marker='.')
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel if xlabel or numericKeys else 'rank')
        self.axes.set_ylabel(ylabel)

    def save(self, oFile: str) -> None:
        """
        Save the graph to a file; the format follows the file extension.

        :param oFile: output file name
        :type oFile: str
        :return: None
        """
        self.figure.savefig(oFile, bbox_inches='tight')


if __name__ == '__main__':
    print("Usage: import plotLineGraphFromDictionary and pass it a dictionary; see the class docstring.")