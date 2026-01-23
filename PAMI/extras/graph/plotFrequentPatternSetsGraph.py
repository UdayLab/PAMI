# plotFrequentPatternSetsGraph is used to visualize the distribution of frequent pattern sets by their size.
#
#  **Importing this algorithm into a python program**
#
#     from PAMI.extras.graph import plotFrequentPatternSetsGraph as fps
#
#     obj = fps.plotFrequentPatternSetsGraph(patterns)
#     obj.plot()
#     obj.save(oFile='fps_graph.png')
#     obj.getStatistics()
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

import matplotlib.pyplot as plt
from typing import Dict, Union


class plotFrequentPatternSetsGraph:
    """
    Visualizes the distribution of frequent pattern sets by their size.

    :param patterns: Dictionary of frequent patterns where keys are tuples of items
    :type patterns: dict
    """

    def __init__(self, patterns: Dict) -> None:
        self.patterns = patterns
        self.patternCounts = self._calculatePatternCounts()

    def _calculatePatternCounts(self) -> Dict[int, int]:
        counts = {}
        for pattern in self.patterns.keys():
            size = len(pattern)
            counts[size] = counts.get(size, 0) + 1
        return dict(sorted(counts.items()))

    def _plot(self) -> None:
        sizes = list(self.patternCounts.keys())
        counts = list(self.patternCounts.values())
        labels = [f"{size}-FPS" for size in sizes]

        plt.figure()
        plt.bar(labels, counts)
        plt.xlabel("Frequent Pattern Set Size")
        plt.ylabel("Count")
        plt.title("Distribution of Frequent Pattern Sets")
        plt.tight_layout()

        for i, count in enumerate(counts):
            plt.text(i, count, str(count), ha='center', va='bottom')

    def plot(self) -> None:
        if not self.patternCounts:
            return
        self._plot()
        plt.show()

    def save(self, oFile: str) -> None:
        if not self.patternCounts:
            return
        self._plot()
        plt.savefig(oFile)
        plt.close()

    def getStatistics(self) -> None:
        if not self.patternCounts:
            return None

        minLen = min(self.patternCounts.keys())
        maxLen = max(self.patternCounts.keys())

        print("Statistics:")
        print(f" Length range: {minLen}-{maxLen}")
        print(" Pattern Size Distribution <size: #count>:")
        for size, count in sorted(self.patternCounts.items()):
            print(f"   {size}:{count}")
        return None


if __name__ == "__main__":
    samplePatterns = {
        ('A',): 110,
        ('B',): 150,
        ('C',): 120,
        ('A', 'B'): 80,
        ('A', 'C'): 70,
        ('B', 'C'): 90,
        ('A', 'B', 'C'): 50,
        ('D',): 110,
        ('A', 'D'): 60,
        ('B', 'D'): 65,
    }

    obj = plotFrequentPatternSetsGraph(samplePatterns)
    obj.plot()
    obj.getStatistics()
