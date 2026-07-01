# plotItemCooccurrenceHeatmap visualizes how often pairs of items occur together.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.graph import plotItemCooccurrenceHeatmap as ch
#
#     obj = ch.plotItemCooccurrenceHeatmap(patterns, maxItems=20)
#
#     obj.plot()
#
#     obj.save(oFile='cooccurrence.png')
#


__copyright__ = """
Copyright (C)  2026 Rage Uday Kiran

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

import warnings
import matplotlib.pyplot as plt
import numpy as np
from urllib.request import urlopen as _urlopen
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple


class plotItemCooccurrenceHeatmap:
    """
    Renders an item-by-item heatmap shaded by pairwise co-occurrence support.

    The input may be either a patterns dict (from any miner's getPatterns()) or
    a transactional database file/URL. Items are coerced to strings internally
    so that integer and string item-IDs are handled uniformly.

    :Attributes:

        iFile : dict or str
            patterns dict {tuple-of-items: support}, or path/URL of a transactional database
        sep : str
            item separator when reading a database file (default tab)
        maxItems : int
            keep only this many of the most frequent items (default 20)

    :Methods:

        plot()
            display the heatmap interactively
        save(oFile)
            save the heatmap to a file
        getMatrix()
            return (items, matrix) pair, computing if necessary
        getStatistics()
            print the strongest co-occurring pair among displayed items

    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

            from PAMI.extras.graph import plotItemCooccurrenceHeatmap as ch

            obj = ch.plotItemCooccurrenceHeatmap(patterns, maxItems=20)

            obj.plot()

            obj.save(oFile='cooccurrence.png')
    """

    def __init__(self, iFile, sep: str = '\t', maxItems: int = 20) -> None:
        """
        :param iFile: patterns dict or path/URL to a transactional database
        :type iFile: dict or str
        :param sep: column separator for database files
        :type sep: str
        :param maxItems: number of top-frequency items to keep
        :type maxItems: int
        """
        if maxItems < 1:
            raise ValueError(f"maxItems must be >= 1, got {maxItems}")
        self._iFile = iFile
        self._sep = sep
        self._maxItems = maxItems
        self._items: List[str] = []
        self._matrix = None

    def __repr__(self) -> str:
        computed = self._matrix is not None
        n = len(self._items) if computed else '?'
        return (f"plotItemCooccurrenceHeatmap(maxItems={self._maxItems}, "
                f"itemsShown={n}, computed={computed})")

    def _pairSupportFromPatterns(self) -> Tuple[Dict[str, int], Dict[frozenset, int]]:
        """Extract single-item supports and pair supports from a patterns dict."""
        singles: Dict[str, int] = {}
        pairs: Dict[frozenset, int] = {}
        for pattern, support in self._iFile.items():
            items = tuple(pattern) if isinstance(pattern, (tuple, list, frozenset)) else (pattern,)
            if len(items) == 1:
                singles[str(items[0])] = support
            elif len(items) == 2:
                pairs[frozenset(str(i) for i in items)] = support
        return singles, pairs

    def _readTransactions(self):
        """Yield transactions (lists of item strings) from a file or URL."""
        if self._iFile.startswith(('http:', 'https:')):
            data = _urlopen(self._iFile)
            for line in data:
                line = line.decode("utf-8").strip()
                yield [i for i in (t.rstrip() for t in line.split(self._sep)) if i]
        else:
            with open(self._iFile, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    yield [i for i in (t.rstrip() for t in line.split(self._sep)) if i]

    def _buildMatrix(self) -> bool:
        """
        Compute self._items (top-N by frequency) and the symmetric
        co-occurrence self._matrix. Returns False when there is no data.
        """
        singles: Dict[str, int] = {}
        pairs: Dict[frozenset, int] = {}

        if isinstance(self._iFile, dict):
            singles, pairs = self._pairSupportFromPatterns()
            if singles:
                freq = singles
            else:
                freq = Counter(i for pair in pairs for i in pair)
                if freq:
                    warnings.warn("No single-item supports in patterns dict; "
                                  "diagonal values will be zero.")
        else:
            freq = Counter()
            for transaction in self._readTransactions():
                freq.update(transaction)
                for a, b in combinations(sorted(set(transaction)), 2):
                    pairs[frozenset((a, b))] = pairs.get(frozenset((a, b)), 0) + 1
            singles = dict(freq)

        if not freq:
            warnings.warn("No items found; cannot build heatmap.")
            return False

        self._items = [item for item, _ in
                       sorted(freq.items(), key=lambda x: -x[1])[:self._maxItems]]
        index = {item: k for k, item in enumerate(self._items)}

        n = len(self._items)
        matrix = np.zeros((n, n), dtype=float)
        for k, item in enumerate(self._items):
            matrix[k][k] = singles.get(item, 0)
        for pair, support in pairs.items():
            a, b = tuple(pair)
            if a in index and b in index:
                matrix[index[a]][index[b]] = support
                matrix[index[b]][index[a]] = support

        self._matrix = matrix
        return True

    def _ensureMatrix(self) -> bool:
        """Build the matrix if it has not been computed yet."""
        if self._matrix is not None:
            return True
        return self._buildMatrix()

    def _renderFigure(self) -> bool:
        """
        Create a matplotlib figure from the current matrix.
        Returns False when there is nothing to draw.
        """
        if not self._ensureMatrix():
            return False

        n = len(self._items)
        plt.figure(figsize=(max(6, n * 0.5), max(5, n * 0.5)))
        im = plt.imshow(self._matrix, cmap='YlOrRd')
        plt.colorbar(im, label='Co-occurrence support')
        plt.xticks(range(n), self._items, rotation=90)
        plt.yticks(range(n), self._items)
        plt.title("Item Co-occurrence Heatmap")
        if n <= 15:
            for i in range(n):
                for j in range(n):
                    value = self._matrix[i][j]
                    if value:
                        plt.text(j, i, int(value), ha='center', va='center', fontsize=7)
        plt.tight_layout()
        return True

    def plot(self) -> None:
        """Display the heatmap interactively."""
        if self._renderFigure():
            plt.show()

    def save(self, oFile: str = 'cooccurrence.png') -> None:
        """
        Save the heatmap to a file.

        :param oFile: output file path
        :type oFile: str
        """
        if self._renderFigure():
            plt.savefig(oFile)
            plt.close()
            print(f"Co-occurrence heatmap saved as {oFile}!")

    def getMatrix(self) -> Tuple[List[str], "np.ndarray"]:
        """Return the (items, matrix) pair, computing it if necessary."""
        self._ensureMatrix()
        return self._items, self._matrix

    def getStatistics(self) -> None:
        """Print the strongest co-occurring pair among the displayed items."""
        items, matrix = self.getMatrix()
        if matrix is None or len(items) == 0:
            print("No items.")
            return
        best, bestPair = 0, None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if matrix[i][j] > best:
                    best, bestPair = matrix[i][j], (items[i], items[j])
        print("Statistics:")
        print(f"  Items shown: {len(items)}")
        if bestPair:
            print(f"  Strongest pair: {bestPair[0]} & {bestPair[1]} (support {int(best)})")


if __name__ == "__main__":
    samplePatterns = {
        ('a',): 6, ('b',): 5, ('c',): 4, ('d',): 3,
        ('a', 'b'): 4, ('a', 'c'): 3, ('b', 'c'): 2, ('c', 'd'): 2, ('a', 'd'): 1,
    }
    obj = plotItemCooccurrenceHeatmap(samplePatterns, maxItems=10)
    obj.save('sampleCooccurrence.png')
    obj.getStatistics()