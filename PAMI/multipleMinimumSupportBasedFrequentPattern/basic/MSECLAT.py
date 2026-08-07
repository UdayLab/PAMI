# MSECLAT is a vertical (tidset) algorithm to discover frequent patterns based on multiple minimum supports in a transactional database.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#             from PAMI.multipleMinimumSupportBasedFrequentPattern.basic import MSECLAT as alg
#
#             obj = alg.MSECLAT(iFile, MIS, sep)
#
#             obj.mine()
#
#             frequentPatterns = obj.getPatterns()
#
#             print("Total number of Frequent Patterns:", len(frequentPatterns))
#
#             obj.save(oFile)
#
#             Df = obj.getPatternsAsDataFrame()
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
     Copyright (C)  2021 Rage Uday Kiran

"""

from PAMI.multipleMinimumSupportBasedFrequentPattern.basic import abstract as _fp
from typing import List, Dict, Tuple, Union
import pandas as pd
from deprecated import deprecated

_fp._sys.setrecursionlimit(20000)


class MSECLAT(_fp._frequentPatterns):
    """
    :Description:   MSECLAT discovers the complete set of frequent patterns based on multiple minimum supports (MIS) in a
                    transactional database using the vertical (tidset) ECLAT strategy. Every item has its own minimum
                    support ``MIS(item)``; a pattern ``P`` is frequent when ``support(P) >= min_{i in P} MIS(i)``. Because
                    that threshold is never below the least MIS in the database (the LMS / minMIS value), the depth-first
                    search over prefix-equivalence classes safely prunes any itemset whose support falls below ``minMIS``
                    (no superset of it can be frequent) while still emitting each itemset that meets its own MIS threshold.
                    It returns the same patterns as CFPGrowth and CFPGrowthPlus.

    :Reference:   Ya-Han Hu and Yen-Liang Chen. 2006. Mining association rules with multiple minimum supports: a new mining
                  algorithm and a support tuning mechanism. Decis. Support Syst. 42, 1 (2006), 1-24.
                  https://doi.org/10.1016/j.dss.2004.09.007

    :param  iFile: str or DataFrame :
                   Name of the Input file to mine complete set of multiple minimum support based frequent patterns
    :param  MIS: str or DataFrame :
                   The multiple minimum support values of the items (a file of ``item<sep>MIS`` lines, or a DataFrame
                   with 'items' and 'MIS' columns).
    :param  oFile: str :
                   Name of the output file to store complete set of frequent patterns
    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default seperator is
                   tab space. However, the users can override their default separator.

    :Attributes:

        startTime : float
            To record the start time of the mining process
        endTime : float
            To record the completion time of the mining process
        finalPatterns : dict
            Storing the complete set of patterns in a dictionary variable
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        Database : list
            To store the transactions of a database in list

    **Executing the code on terminal:**
    -------------------------------------
    .. code-block:: console

       Format:

      (.venv) $ python3 MSECLAT.py <inputFile> <outputFile> <MISFile>

      Examples:

      (.venv) $ python3 MSECLAT.py sampleDB.txt patterns.txt MISFile.txt

    .. note:: MIS values are considered in support count.

    **Sample run of the importing code:**
    ----------------------------------------
    .. code-block:: python

            from PAMI.multipleMinimumSupportBasedFrequentPattern.basic import MSECLAT as alg

            obj = alg.MSECLAT(iFile, MIS, sep)

            obj.mine()

            frequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPatterns))

            obj.save(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

    **Credits:**
    --------------
        The complete program was written under the supervision of Professor Rage Uday Kiran.

    """

    _startTime = float()
    _endTime = float()
    _MIS = str
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []

    def __init__(self, iFile, MIS, sep='\t') -> None:
        super().__init__(iFile, MIS, sep)
        self._MISValues = {}

    def _creatingItemSets(self) -> None:
        """
        Storing the complete transactions of the database/input file in a database variable
        """
        self._Database = []
        if isinstance(self._iFile, _fp._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
                self._Database = [x.split(self._sep) if isinstance(x, str) else list(x) for x in self._Database]
        if isinstance(self._iFile, str):
            if _fp._validators.url(self._iFile):
                data = _fp._urlopen(self._iFile)
                for line in data:
                    line = line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def _getMISValues(self) -> None:
        """
        Storing the minimum supports given by the user for each item in the database.
        """
        self._MISValues = {}
        if isinstance(self._MIS, _fp._pd.DataFrame):
            items, mis = [], []
            if self._MIS.empty:
                print("its empty..")
            i = self._MIS.columns.values.tolist()
            if 'items' in i:
                items = self._MIS['items'].tolist()
            if 'MIS' in i:
                mis = self._MIS['MIS'].tolist()
            for i in range(len(items)):
                self._MISValues[items[i]] = int(mis[i])
        if isinstance(self._MIS, str):
            if _fp._validators.url(self._MIS):
                data = _fp._urlopen(self._MIS)
                for line in data:
                    line = line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._MISValues[temp[0]] = int(temp[1])
            else:
                try:
                    with open(self._MIS, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._MISValues[temp[0]] = int(temp[1])
                except IOError:
                    print("File Not Found")
                    quit()

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self) -> None:
        """
        main program to start the operation
        """
        self.mine()

    def _threshold(self, itemSet: Tuple[str, ...]) -> int:
        """
        The frequency threshold of ``itemSet`` under multiple minimum supports: the minimum MIS among its items.
        """
        return min(self._MISValues[i] for i in itemSet)

    def _recursive(self, members: List[Tuple[Tuple[str, ...], set]], minMIS: int) -> None:
        """
        Depth-first ECLAT over a prefix-equivalence class. Member ``i`` is extended with the last item of each later
        member ``j``; the extension's tidset is the intersection of the two tidsets. An itemset is kept for further
        extension whenever its support reaches ``minMIS`` (below that no superset can be frequent), and it is emitted
        only when its support meets its own multiple-minimum-support threshold.
        """
        for i in range(len(members)):
            children: List[Tuple[Tuple[str, ...], set]] = []
            for j in range(i + 1, len(members)):
                candidate = members[i][0] + (members[j][0][-1],)
                tidset = members[i][1] & members[j][1]
                support = len(tidset)
                if support >= minMIS:
                    children.append((candidate, tidset))
                    if support >= self._threshold(candidate):
                        self._finalPatterns["\t".join(candidate)] = support
            if len(children) > 1:
                self._recursive(children, minMIS)

    def mine(self) -> None:
        """
        main program to start the operation
        """
        self._startTime = _fp._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        self._finalPatterns = {}
        self._creatingItemSets()
        self._getMISValues()

        minMIS = min(self._MISValues.values())

        # Vertical representation: tidset of every item.
        tidsets: Dict[str, set] = {}
        for tid, transaction in enumerate(self._Database):
            for item in transaction:
                if item not in tidsets:
                    tidsets[item] = set()
                tidsets[item].add(tid)

        # Least-minimum-support pruning: an item can only appear in a frequent pattern if its support >= minMIS.
        candidates = {item: len(tid) for item, tid in tidsets.items()
                      if item in self._MISValues and len(tid) >= minMIS}
        # Support-ascending total order (with item tiebreaker) - the ECLAT processing order.
        orderedItems = sorted(candidates.keys(), key=lambda it: (candidates[it], it))

        members: List[Tuple[Tuple[str, ...], set]] = []
        for item in orderedItems:
            if candidates[item] >= self._MISValues[item]:
                self._finalPatterns[item] = candidates[item]
            members.append(((item,), tidsets[item]))
        if len(members) > 1:
            self._recursive(members, minMIS)

        print("Frequent patterns were generated successfully using MSECLAT algorithm")
        self._endTime = _fp._time.time()
        process = _fp._psutil.Process(_fp._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss

    def getMemoryUSS(self) -> float:
        """
        Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process
        :rtype: float
        """
        return self._memoryUSS

    def getMemoryRSS(self) -> float:
        """
        Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """
        return self._memoryRSS

    def getRuntime(self) -> float:
        """
        Calculating the total amount of runtime taken by the mining process

        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """
        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self) -> pd.DataFrame:
        """
        Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """
        dataframe = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a.replace('\t', ' '), b])
            dataframe = _fp._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataframe

    def save(self, outFile: str) -> None:
        """
        Complete set of frequent patterns will be loaded in to an output file

        :param outFile: name of the output file
        :type outFile: file
        :return: None
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self) -> Dict[str, int]:
        """
        Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self) -> None:
        """
        this function is used to print the results
        """
        print("Total number of  Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_fp._sys.argv) == 4 or len(_fp._sys.argv) == 5:
        if len(_fp._sys.argv) == 5:
            _ap = MSECLAT(_fp._sys.argv[1], _fp._sys.argv[3], _fp._sys.argv[4])
        if len(_fp._sys.argv) == 4:
            _ap = MSECLAT(_fp._sys.argv[1], _fp._sys.argv[3])
        _ap.mine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_fp._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
