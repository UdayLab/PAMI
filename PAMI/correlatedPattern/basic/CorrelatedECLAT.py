# CorrelatedECLAT is a vertical algorithm to discover correlated patterns in a transactional database.
#
# **Importing this algorithm into a python program**
#
#             from PAMI.correlatedPattern.basic import CorrelatedECLAT as alg
#
#             iFile = 'sampleTDB.txt'
#
#             minSup = 0.25 # can be specified between 0 and 1
#
#             minAllConf = 0.2 # can  be specified between 0 and 1
#
#             obj = alg.CorrelatedECLAT(iFile, minSup, minAllConf, sep)
#
#             obj.mine()
#
#             patterns = obj.getPatterns()
#
#             print("Total number of  Patterns:", len(patterns))
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

from PAMI.correlatedPattern.basic import abstract as _ab
import pandas as _pd
from typing import List, Dict, Tuple, Union
from deprecated import deprecated

_ab._sys.setrecursionlimit(20000)


class CorrelatedECLAT(_ab._correlatedPatterns):
    """
    **About this algorithm**

    :**Description**: CorrelatedECLAT discovers the complete set of correlated patterns in a transactional database using
                      the vertical (tidset) ECLAT strategy. A pattern is correlated when its support is at least ``minSup``
                      and its all-confidence ``support(P) / max_{i in P} support(i)`` is at least ``minAllConf``. Both
                      support and all-confidence are anti-monotone, so the depth-first search over prefix-equivalence
                      classes prunes any itemset that is not itself correlated. It returns the same patterns as CoMine
                      and CoMinePlus.

    :**Reference**: Omiecinski, E. R. (2003). Alternative interest measures for mining associations in databases. IEEE TKDE 15(1), 57-69.
                    Lee, Y.K., Kim, W.Y., Cao, D., Han, J. (2003). CoMine: efficient mining of correlated patterns. In ICDM (pp. 581-584).

    :**parameters**:    - **iFile** (*str or DataFrame or URL*) -- *Name of the Input file to mine complete set of correlated patterns*
                        - **oFile** (*str*) -- *Name of the output file to store complete set of correlated patterns*
                        - **minSup** (*int or float or str*) -- *The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.*
                        - **minAllConf** (*float*) -- *The user can specify minAllConf values within the range (0, 1).*
                        - **sep** (*str*) -- *This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.*

    :**Attributes**:    - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*
                        - **startTime** (*float*) -- *To record the start time of the mining process.*
                        - **endTime** (*float*) -- *To record the completion time of the mining process.*
                        - **finalPatterns** (*dict*) -- *it represents to store the patterns.*

    **Execution methods**

    **Terminal command**

    .. code-block:: console

      Format:

      (.venv) $ python3 CorrelatedECLAT.py <inputFile> <outputFile> <minSup> <minAllConf> <sep>

      Example Usage:

      (.venv) $ python3 CorrelatedECLAT.py sampleTDB.txt output.txt 0.25 0.2

    .. note:: minSup can be specified in support count or a value between 0 and 1.

    **Calling from a python program**

    .. code-block:: python

            from PAMI.correlatedPattern.basic import CorrelatedECLAT as alg

            iFile = 'sampleTDB.txt'

            minSup = 0.25 # can be specified between 0 and 1

            minAllConf = 0.2 # can  be specified between 0 and 1

            obj = alg.CorrelatedECLAT(iFile, minSup, minAllConf, sep)

            obj.mine()

            patterns = obj.getPatterns()

            print("Total number of  Patterns:", len(patterns))

            obj.save(oFile)

            df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

    **Credits**

    The complete program was written under the supervision of Professor Rage Uday Kiran.

    """

    _startTime = float()
    _endTime = float()
    _minSup = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _minAllConf = 0.0
    _Database = []
    _mapSupport = {}
    _sep = "\t"

    def __init__(self, iFile: Union[str, _pd.DataFrame], minSup: Union[int, float, str], minAllConf: float, sep: str = "\t") -> None:
        super().__init__(iFile, minSup, minAllConf, sep)

    def _creatingItemSets(self) -> None:
        """
        Storing the complete transactions of the database/input file in a database variable
        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
                self._Database = [x.split(self._sep) for x in self._Database]
            else:
                print("The column name should be Transactions and each line should be separated by tab space or a seperator specified by the user")
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def _convert(self, value: Union[int, float, str]):
        """
        To convert the type of user specified minSup value

        :param value: user specified minSup value
        :type value: int or float or str
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self._Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self._Database) * value)
            else:
                value = int(value)
        return value

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self) -> None:
        """
        main method to start
        """
        self.mine()

    def _maxSup(self, itemSet: Tuple[str, ...]) -> int:
        """
        Maximum single-item support among the items of ``itemSet`` (the all-confidence denominator).
        """
        return max(self._mapSupport[i] for i in itemSet)

    def _recursive(self, members: List[Tuple[Tuple[str, ...], set]]) -> None:
        """
        Depth-first ECLAT over a prefix-equivalence class. Member ``i`` is extended with the last item of each later
        member ``j``; the extension's tidset is the intersection of the two tidsets and its all-confidence is
        ``support / maxSup(items)``. Only correlated itemsets are extended (support and all-confidence are anti-monotone).
        """
        for i in range(len(members)):
            children: List[Tuple[Tuple[str, ...], set]] = []
            for j in range(i + 1, len(members)):
                candidate = members[i][0] + (members[j][0][-1],)
                tidset = members[i][1] & members[j][1]
                support = len(tidset)
                if support >= self._minSup:
                    allConf = support / self._maxSup(candidate)
                    if allConf >= self._minAllConf:
                        self._finalPatterns[candidate] = [support, allConf]
                        children.append((candidate, tidset))
            if len(children) > 1:
                self._recursive(children)

    def mine(self) -> None:
        """
        main method to start
        """
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        self._finalPatterns = {}
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)

        # Build the vertical representation: tidset (set of transaction ids) of every item.
        tidsets: Dict[str, set] = {}
        for tid, transaction in enumerate(self._Database):
            for item in transaction:
                if item not in tidsets:
                    tidsets[item] = set()
                tidsets[item].add(tid)

        self._mapSupport = {k: len(v) for k, v in tidsets.items() if len(v) >= self._minSup}
        # Support-ascending total order (with item tiebreaker) - the ECLAT processing order.
        frequentItems = sorted(self._mapSupport.keys(), key=lambda it: (self._mapSupport[it], it))

        members: List[Tuple[Tuple[str, ...], set]] = []
        for item in frequentItems:
            self._finalPatterns[(item,)] = [self._mapSupport[item], 1.0]
            members.append(((item,), tidsets[item]))
        if len(members) > 1:
            self._recursive(members)

        print("Correlated patterns were generated successfully using CorrelatedECLAT algorithm")
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
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

    def getPatternsAsDataFrame(self) -> _pd.DataFrame:
        """
        Storing final correlated patterns in a dataframe

        :return: returning correlated patterns in a dataframe
        :rtype: pd.DataFrame
        """
        dataframe = {}
        data = []
        for a, b in self._finalPatterns.items():
            pat = " "
            for i in a:
                pat += str(i) + " "
            data.append([pat, b[0], b[1]])
            dataframe = _ab._pd.DataFrame(data, columns=['Patterns', 'Support', 'Confidence'])
        return dataframe

    def save(self, outFile) -> None:
        """
        Complete set of correlated patterns will be saved into an output file

        :param outFile: name of the outputfile
        :type outFile: file
        :return: None
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            pat = ""
            for i in x:
                pat += str(i) + "\t"
            patternsAndSupport = pat.strip() + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % patternsAndSupport)

    def getPatterns(self) -> Dict[Tuple[str, ...], List[Union[int, float]]]:
        """
        Function to send the set of correlated patterns after completion of the mining process

        :return: returning correlated patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self) -> None:
        """
        function to print the result after completing the process

        :return: None
        """
        print("Total number of Correlated Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = CorrelatedECLAT(_ab._sys.argv[1], _ab._sys.argv[3], float(_ab._sys.argv[4]), _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = CorrelatedECLAT(_ab._sys.argv[1], _ab._sys.argv[3], float(_ab._sys.argv[4]))
        _ap.mine()
        print("Total number of Correlated-Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in seconds:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
