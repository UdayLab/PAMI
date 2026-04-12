# FPGrowthbitset is one of the fundamental algorithm to discover frequent patterns in a transactional database.
#
# **Importing this algorithm into a python program**
#
#             from PAMI.frequentPattern.basic.FPGrowthbitset import FPGrowthbitset as alg
#
#             iFile = 'sampleDB.txt'
#
#             minSup = 10  # can also be specified between 0 and 1
#
#             obj = alg.FPGrowthbitset(iFile, minSup)
#
#             obj.mine()
#
#             frequentPatterns = obj.getPatterns()
#
#             print("Total number of Frequent Patterns:", len(frequentPatterns))
#
#             obj.save(oFile)
#
#             Df = obj.getPatternInDataFrame()
#
#             memUSS = obj.getMemoryUSS()
#
#             print("Total Memory in USS:", memUSS)
#
#             memRSS = obj.getMemoryRSS()
#
#             print("Total Memory in RSS", memRSS)
#
#             run = obj.getRuntime()
#
#             print("Total ExecutionTime in seconds:", run)
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

from PAMI.frequentPattern.basic import abstract as _ab
from deprecated import deprecated
from typing import Dict, Any, Union, List, Tuple
from collections import defaultdict, Counter
import numpy as _np
import math as _math
import gc as _gc


class FPGrowthbitset(_ab._frequentPatterns):
    """
    **About this algorithm**

    :**Description**:   FPGrowthbitset is one of the fundamental algorithm to discover frequent patterns in a transactional database.

    :**Reference**:     Mohammed Javeed Zaki: Scalable Algorithms for Association Mining. IEEE Trans. Knowl. Data Eng. 12(3):
                        372-390 (2000), https://ieeexplore.ieee.org/document/846291

    :**Parameters**:    - **iFile** (*str or URL or dataFrame*) -- *Name of the Input file to mine complete set of frequent patterns.*
                        - **oFile** (*str*) -- *Name of the output file to store complete set of frequent patterns.*
                        - **minSup** (*int or float or str*) -- *The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.*
                        - **sep** (*str*) -- *This variable is used to distinguish items from one another in a transaction. The default separator is tab space. However, the users can override their default separator.*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the mining process.*
                        - **endTime** (*float*) -- *To record the completion time of the mining process.*
                        - **finalPatterns** (*dict*) -- *Storing the complete set of patterns in a dictionary variable.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*
                        - **Database** (*list*) -- *To store the transactions of a database in list.*


    **Execution methods**

    **Terminal command**

    .. code-block:: console

      Format:

      (.venv) $ python3 FPGrowthbitset.py <inputFile> <outputFile> <minSup>

      Example Usage:

      (.venv) $ python3 FPGrowthbitset.py sampleDB.txt patterns.txt 10.0

    .. note:: minSup can be specified in support count or a value between 0 and 1.


    **Calling from a python program**

    .. code-block:: python

            import PAMI.frequentPattern.basic.FPGrowthbitset as alg

            iFile = 'sampleDB.txt'

            minSup = 10  # can also be specified between 0 and 1

            obj = alg.FPGrowthbitset(iFile, minSup)

            obj.mine()

            frequentPattern = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPattern))

            obj.save(oFile)

            Df = obj.getPatternInDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

    """

    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _minSup = str()
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _bitsets = {}

    def _creatingItemSets(self) -> None:
        """
        Storing the complete transactions of the database/input file in a database variable
        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
                return
            cols = self._iFile.columns.values.tolist()
            if 'Transactions' in cols:
                self._Database = [x.split(self._sep) for x in self._iFile['Transactions'].tolist()]
            else:
                arr = self._iFile.values
                col_names = self._iFile.columns.tolist()
                for row in arr:
                    self._Database.append([col_names[j] for j, val in enumerate(row) if val])

        elif isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    temp = [x for x in line.decode("utf-8").strip().split(self._sep) if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            splitter = [x for x in line.strip().split(self._sep) if x]
                            self._Database.append(splitter)
                except IOError:
                    print("File Not Found")

    def _convert(self, value: Union[int, float, str], dbLen: int) -> int:
        if type(value) is int:
            return value
        if type(value) is float:
            return _math.ceil(dbLen * value)
        if type(value) is str:
            if '.' in value:
                return _math.ceil(dbLen * float(value))
            return int(value)
        return int(value)

    def _bitPacker(self, tids: List[int], total: int) -> int:
        packed = 0
        for tid in tids:
            packed |= (1 << (total - tid - 1))
        return packed

    def _getBitCount(self, n: int) -> int:
        if hasattr(n, 'bit_count'):
            return n.bit_count()
        return bin(n).count('1')

    def _dfs(self, items: List[Any], current_bitset: int, prefix: List[Any], minSup: int):
        """
        Depth-First Search using bitwise-AND intersections.
        """
        for i in range(len(items)):
            item = items[i]
            intersection = current_bitset & self._bitsets[item]
            count = self._getBitCount(intersection)
            
            if count >= minSup:
                new_prefix = prefix + [item]
                self._finalPatterns[tuple(sorted(new_prefix))] = count
                
                # Recurse with remaining items (depth-first)
                if i + 1 < len(items):
                    self._dfs(items[i+1:], intersection, new_prefix, minSup)

    def startMine(self) -> None:
        """
        Frequent pattern mining process will start from here
        """
        self.mine()

    def mine(self) -> None:
        """
        Starts the mining process. Automatically selects the most efficient path.
        """
        self._startTime = _ab._time.time()
        self._finalPatterns = {}
        self._bitsets = {}
        
        # --- Detect Mode & Build Bitsets ---
        is_binary_df = False
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if 'Transactions' not in self._iFile.columns:
                is_binary_df = True

        if is_binary_df:
            df = self._iFile
            dbLen = len(df)
            minSup = self._convert(self._minSup, dbLen)
            
            arr = df.values
            col_sums = arr.sum(axis=0)
            frequent_cols = _np.where(col_sums >= minSup)[0]
            
            # Pack bitsets using NumPy indices
            for col_idx in frequent_cols:
                tids = _np.where(arr[:, col_idx])[0]
                item_name = df.columns[col_idx]
                self._bitsets[item_name] = self._bitPacker(tids.tolist(), dbLen)
            
            frequent_items = sorted(self._bitsets.keys(), key=lambda x: self._getBitCount(self._bitsets[x]), reverse=True)
        else:
            self._creatingItemSets()
            dbLen = len(self._Database)
            minSup = self._convert(self._minSup, dbLen)
            
            tid_lists = defaultdict(list)
            for tid, t in enumerate(self._Database):
                for item in t:
                    tid_lists[item].append(tid)
            
            for item, tids in tid_lists.items():
                if len(tids) >= minSup:
                    self._bitsets[item] = self._bitPacker(tids, dbLen)
            
            del self._Database
            frequent_items = sorted(self._bitsets.keys(), key=lambda x: self._getBitCount(self._bitsets[x]), reverse=True)

        # --- Depth First Search ---
        # Seed the DFS with 1-itemsets
        all_bits = (1 << dbLen) - 1
        self._dfs(frequent_items, all_bits, [], minSup)

        _gc.collect()

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using FPGrowthbitset algorithm")

    def getMemoryUSS(self) -> float:
        """
        Total amount of USS memory consumed by the program will be retrieved from this function
        """
        return self._memoryUSS

    def getMemoryRSS(self) -> float:
        """
        Total amount of RSS memory consumed by the program will be retrieved from this function
        """
        return self._memoryRSS

    def getRuntime(self) -> float:
        """
        Total amount of runtime taken by the program will be retrieved from this function
        """
        return self._endTime - self._startTime

    def getPatterns(self) -> Dict[tuple, int]:
        """
        Complete set of frequent patterns generated will be retrieved from this function
        """
        return self._finalPatterns
    
    def getPatternsAsDataFrame(self) -> _ab._pd.DataFrame:
        """
        Complete set of frequent patterns will be loaded in to data frame from this function
        """
        dataFrame = _ab._pd.DataFrame(list([[self._sep.join(x), y] for x,y in self._finalPatterns.items()]), columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, outFile: str, seperator: str = "\t") -> None:
        """
        Complete set of frequent patterns will be saved in to an output file from this function
        :param outFile: Name of the output file
        :type outFile: str
        """
        with open(outFile, 'w') as f:
            for x, y in self._finalPatterns.items():
                f.write(f"{seperator.join(x)}:{y}\n")

    def printResults(self) -> None:
        """
        This function is used to print the result
        """
        print("Total number of Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS:", self.getMemoryRSS())
        print("Total ExecutionTime in seconds:", self.getRuntime())


if __name__ == "__main__":
    # python3 FPGrowthbitset.py <input> <output> <minSup>
    import sys
    if len(sys.argv) >= 4:
        if len(sys.argv) == 5:
            _ap = FPGrowthbitset(sys.argv[1], sys.argv[3], sys.argv[4])
        else:
            _ap = FPGrowthbitset(sys.argv[1], sys.argv[3])
        _ap.mine()
        _ap.printResults()
        _ap.save(sys.argv[2])
    else:
        print("Format: python3 FPGrowthbitset.py <input> <output> <minSup>")
