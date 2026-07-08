# GenMax is an algorithm to discover maximal frequent patterns in a transactional database.
#
# **Importing this algorithm into a python program**
#--------------------------------------------------------------------
#             from PAMI.frequentPattern.maximal import GenMax as alg
#
#             obj = alg.GenMax(iFile, minSup)
#
#             obj.mine()
#
#             maximalPatterns = obj.getPatterns()
#
#             print("Total number of Maximal Frequent Patterns:", len(maximalPatterns))
#
#             obj.save(oFile)
#
#             memUSS = obj.getMemoryUSS()
#
#             print("Total Memory in USS:", memUSS)
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

from PAMI.frequentPattern.maximal import abstract as _ab
from deprecated import deprecated


class GenMax(_ab._frequentPatterns):
    """
    :**Description**:   GenMax discovers the complete set of maximal frequent itemsets in a
                        transactional database. A frequent itemset is *maximal* if none of its
                        proper supersets is frequent. GenMax explores the itemset lattice with a
                        backtracking depth-first search over vertical tid-sets and prunes any
                        branch whose head-union-tail is already contained in a discovered
                        maximal itemset.

    :**Reference**:     Karam Gouda and Mohammed J. Zaki. GenMax: An Efficient Algorithm for
                        Mining Maximal Frequent Itemsets. Data Mining and Knowledge Discovery
                        11, 223-242 (2005). https://doi.org/10.1007/s10618-005-0002-x

    :**Parameters**:    - **iFile** (*str or URL or dataFrame*) -- *Name of the input file to mine the maximal frequent patterns.*
                        - **oFile** (*str*) -- *Name of the output file to store the maximal frequent patterns.*
                        - **minSup** (*int or float or str*) -- *The user can specify minSup in count or as a proportion of the database size.*
                        - **sep** (*str*) -- *Separator used to distinguish items within a transaction. Default is a tab space.*

    **Execution methods**

    **Terminal command**

    .. code-block:: console

      Format:

      (.venv) $ python3 GenMax.py <inputFile> <outputFile> <minSup>

      Example Usage:

      (.venv) $ python3 GenMax.py sampleDB.txt patterns.txt 10.0

    .. note:: minSup can be specified in support count or a value between 0 and 1.


    **Calling from a python program**

    .. code-block:: python

            from PAMI.frequentPattern.maximal import GenMax as alg

            obj = alg.GenMax('sampleDB.txt', 10)

            obj.mine()

            maximalPatterns = obj.getPatterns()

            print("Total number of Maximal Frequent Patterns:", len(maximalPatterns))

            obj.save('output.txt')
 """

    _startTime = float()
    _endTime = float()
    _minSup = str()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []

    def __init__(self, iFile, minSup, sep="\t"):
        super().__init__(iFile, minSup, sep)

    def _creatingItemSets(self):
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
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def _convert(self, value):
        """
        To convert the type of user specified minSup value
        :param value: user specified minSup value
        :type value: int or float or str
        :return: converted type minSup value
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

    def _addToMFI(self, itemset, support, mfi):
        """
        Insert ``itemset`` into the maximal-frequent-itemset antichain ``mfi``, keeping only
        maximal members: skip if already subsumed, else drop any member it subsumes.

        :param itemset: candidate maximal itemset
        :type itemset: frozenset
        :param support: support of the itemset
        :type support: int
        :param mfi: current maximal itemsets (frozenset -> support)
        :type mfi: dict
        """
        for m in mfi:
            if itemset <= m:
                return
        for m in [m for m in mfi if m <= itemset]:
            del mfi[m]
        mfi[itemset] = support

    def _backtrackDiffset(self, prefix, combineSet, mfi):
        """
        Backtracking DFS over the **diffset** lattice (dECLAT) -- used on dense data.

        Each candidate carries its diffset relative to ``prefix``; a child's diffset is
        the set difference of two sibling diffsets, and support drops by the child
        diffset's size. Diffsets stay small on dense data, where tid-set intersection is
        most costly.

        :param combineSet: candidate extensions as (item, support, diffSet) triples
        """
        for i in range(len(combineSet)):
            itemI, supI, diffI = combineSet[i]
            mI = prefix + [itemI]

            newCombine = []
            for j in range(i + 1, len(combineSet)):
                itemJ, supJ, diffJ = combineSet[j]
                diffIJ = diffJ - diffI               # dECLAT diffset recurrence
                supIJ = supI - len(diffIJ)
                if supIJ >= self._minSup:
                    newCombine.append((itemJ, supIJ, diffIJ))

            if not newCombine:
                self._addToMFI(frozenset(mI), supI, mfi)
            else:
                hut = frozenset(mI).union(c[0] for c in newCombine)
                if any(hut <= m for m in mfi):
                    continue
                self._backtrackDiffset(mI, newCombine, mfi)

    def _backtrackTidset(self, prefix, combineSet, mfi):
        """
        Backtracking DFS over the **tid-set** lattice (ECLAT) -- used on sparse data,
        where tid-sets are smaller than diffsets.

        :param combineSet: candidate extensions as (item, tidSet) pairs
        """
        for i in range(len(combineSet)):
            itemI, tidI = combineSet[i]
            mI = prefix + [itemI]

            newCombine = []
            for j in range(i + 1, len(combineSet)):
                itemJ, tidJ = combineSet[j]
                inter = tidI & tidJ
                if len(inter) >= self._minSup:
                    newCombine.append((itemJ, inter))

            if not newCombine:
                self._addToMFI(frozenset(mI), len(tidI), mfi)
            else:
                hut = frozenset(mI).union(c[0] for c in newCombine)
                if any(hut <= m for m in mfi):
                    continue
                self._backtrackTidset(mI, newCombine, mfi)

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self):
        """
        Mining process will start from this function
        """
        self.mine()

    def mine(self):
        """
        Mining process will start from this function
        """
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._minSup is None:
            raise Exception("Please enter the Minimum Support")
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        self._finalPatterns = {}

        # Build vertical tid-sets for every item.
        tidSets = {}
        for tid, transaction in enumerate(self._Database):
            for item in transaction:
                if item not in tidSets:
                    tidSets[item] = set()
                tidSets[item].add(tid)

        n = len(self._Database)
        frequent = [(item, tids) for item, tids in tidSets.items() if len(tids) >= self._minSup]

        mfi = {}
        dense = frequent and (sum(len(t) for _, t in frequent) / len(frequent)) >= n / 2
        if dense:
            allTids = set(range(n))
            combineSet = [(item, len(tids), allTids - tids) for item, tids in frequent]
            combineSet.sort(key=lambda x: x[1])          # increasing support (GenMax ordering)
            self._backtrackDiffset([], combineSet, mfi)
        else:
            combineSet = [(item, tids) for item, tids in frequent]
            combineSet.sort(key=lambda x: len(x[1]))
            self._backtrackTidset([], combineSet, mfi)

        for itemset, support in mfi.items():
            key = "\t".join(sorted(itemset))
            self._finalPatterns[key] = support

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Maximal Frequent patterns were generated successfully using GenMax algorithm")

    def getMemoryUSS(self):
        """
        Total amount of USS memory consumed by the mining process will be retrieved from this function
        :return: returning USS memory consumed by the mining process
        :rtype: float
        """
        return self._memoryUSS

    def getMemoryRSS(self):
        """
        Total amount of RSS memory consumed by the mining process will be retrieved from this function
        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """
        return self._memoryRSS

    def getRuntime(self):
        """
        Calculating the total amount of runtime taken by the mining process
        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """
        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self):
        """
        Storing final maximal frequent patterns in a dataframe
        :return: returning maximal frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """
        dataframe = _ab._pd.DataFrame(
            list([[x.replace('\t', ' '), y] for x, y in self._finalPatterns.items()]),
            columns=['Patterns', 'Support'])
        return dataframe

    def save(self, outFile):
        """
        Complete set of maximal frequent patterns will be loaded in to an output file
        :param outFile: name of the output file
        :type outFile: csvfile
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of maximal frequent patterns after completion of the mining process
        :return: returning maximal frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        """
        This function is used to print the results
        """
        print('Total number of Maximal Frequent Patterns: ' + str(len(self.getPatterns())))
        print('Runtime: ' + str(self.getRuntime()))
        print('Memory (RSS): ' + str(self.getMemoryRSS()))
        print('Memory (USS): ' + str(self.getMemoryUSS()))


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = GenMax(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = GenMax(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.mine()
        _ap.save(_ab._sys.argv[2])
        print("Total number of Maximal Frequent Patterns:", len(_ap.getPatterns()))
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
