# topKPatterns is used to find top k length patterns in input file.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.syntheticDataGenerator import topKPatterns as tK
#
#     obj = tK.topKPatterns(" ", 10, "\t")
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


class topKPatterns:
    """
    Description:
        find top k length patterns in input file.

    Attributes:
    -----------
        inputFile : str
            input file name or path
        k : int
            rank of pattern length. default is 10
        sep : str
            separator which separate patterns in input file. default is tab space

    Methods:
    -------
        getTopKPatterns()
            return top k patterns as dict
        storeTopKPatterns(outputFile)
            store top k patterns into output file.

        **Importing this algorithm into a python program**
        --------------------------------------------------------
        .. code-block:: python

        from PAMI.extras.syntheticDataGenerator import topKPatterns as plt

        obj = plt.scatterPlotSpatialPoints(" ", 10, "\t" )

        obj.save()
    """
    def __init__(self, inputFile: str, k: int=10, sep: str='\t') -> None:
        self.inputFile = inputFile
        self.k = k
        self.sep = sep

    def getTopKPatterns(self) -> dict:
        """
        get top k length patterns. user can defined k value.
        :return: top k length patterns as dictionary. top k patterns = {patternId: pattern}
        """
        with open(self.inputFile, 'r') as f:
            patterns = [[item for item in line.strip().split(':')][0].split(self.sep)[:-1] for line in f]
        patterns = sorted(patterns, key=lambda x: len(x[0]), reverse=True)
        return {patternId: patterns[patternId - 1] for patternId in range(1, int(self.k)+1)}

    def save(self, outputFile: str) -> None:
        """
        store top k length patterns into file. user can defined k value.
        :param outputFile: output file name or path
        :type outputFile: str
        """
        with open(self.inputFile, 'r') as f:
            patterns = [[item for item in line.strip().split(':')][0].split(self.sep)[:-1] for line in f]
            patterns = sorted(patterns, key=lambda x: len(x[0]), reverse=True)
        with open(outputFile, 'w') as f:
            patternId = 1
            for pattern in patterns[:self.k]:
                for item in pattern:
                    f.write(f'{patternId}\t{item}\n')
