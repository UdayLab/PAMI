# PFPGrowth is one of the fundamental algorithm to discover periodic-frequent patterns in a transactional database.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#
#             from PAMI.periodicFrequentPattern.basic import PFPGrowth as alg
#
#             obj = alg.PFPGrowth(iFile, minSup, maxPer)
#
#             obj.startMine()
#
#             periodicFrequentPatterns = obj.getPatterns()
#
#             print("Total number of Periodic Frequent Patterns:", len(periodicFrequentPatterns))
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

from PAMI.periodicFrequentPattern.basic import abstract as _ab
from typing import List, Dict, Tuple, Set, Union, Any, Generator

import pandas as pd
from deprecated import deprecated
import numpy as np

_maxPer = float()
_minSup = float()
_lno = int()


class _Node(object):
    """
    A class used to represent the node of frequentPatternTree

    :Attributes:

        item : int or None
            Storing item of a node
        timeStamps : list
            To maintain the timestamps of a database at the end of the branch
        parent : node
            To maintain the parent of every node
        children : list
            To maintain the children of a node

    :Methods:

        addChild(itemName)
            Storing the children to their respective parent nodes
        """

    def __init__(self, item, locations, parent=None):
        self.item = item
        self.locations = locations
        self.parent = parent
        self.children = {}

    def addChild(self, item, locations):
        if item not in self.children:
            self.children[item] = _Node(item, locations, self)
        else:
            self.children[item].locations = locations + self.children[item].locations
            
        return self.children[item]

    def traverse(self):
        transaction = []
        locs = self.locations
        node = self.parent
        while node.parent is not None:
            transaction.append(node.item)
            node = node.parent
        return transaction[::-1], locs

    def traverse(self):
        transaction = []
        locs = self.locations
        node = self.parent
        while node.parent is not None:
            transaction.append(node.item)
            node = node.parent
        return transaction[::-1], locs

class PFPGrowth(_ab._periodicFrequentPatterns):
    """
    :Description:   PFPGrowth is one of the fundamental algorithm to discover periodic-frequent patterns in a transactional database.

    :Reference:   Syed Khairuzzaman Tanbeer, Chowdhury Farhan, Byeong-Soo Jeong, and Young-Koo Lee, "Discovering Periodic-Frequent
                   Patterns in Transactional Databases", PAKDD 2009, https://doi.org/10.1007/978-3-642-01307-2_24

    :param  iFile: str :
                   Name of the Input file to mine complete set of periodic frequent pattern's
    :param  oFile: str :
                   Name of the output file to store complete set of periodic frequent pattern's
    :param  minSup: str:
                   Controls the minimum number of transactions in which every item must appear in a database.
    :param  maxPer: float:
                   Controls the maximum number of transactions in which any two items within a pattern can reappear.
    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.


    :Attributes:

        iFile : file
            Name of the Input file or path of the input file
        oFile : file
            Name of the output file or path of the output file
        minSup : int or float or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        maxPer : int or float or str
            The user can specify maxPer either in count or proportion of database size.
            If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxPer=10 will be treated as integer, while maxPer=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
            However, the users can override their default separator.
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        Database : list
            To store the transactions of a database in list
        mapSupport : Dictionary
            To maintain the information of item and their frequency
        lno : int
            To represent the total no of transaction
        tree : class
            To represents the Tree class
        itemSetCount : int
            To represents the total no of patterns
        finalPatterns : dict
            To store the complete patterns

    :Methods:

        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        save(oFile)
            Complete set of periodic-frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of periodic-frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        creatingItemSets(fileName)
            Scans the dataset and stores in a list format
        PeriodicFrequentOneItem()
            Extracts the one-periodic-frequent patterns from database
        updateDatabases()
            Update the database by removing aperiodic items and sort the Database by item decreased support
        buildTree()
            After updating the Database, remaining items will be added into the tree by setting root node as null
        convert()
            to convert the user specified value



    **Credits:**
    --------------
             The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.
    """
    _startTime = float()
    _endTime = float()
    _minSup = str()
    _maxPer = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _rank = {}
    _rankedUp = {}
    _lno = 0

    def _creatingItemSets(self) -> None:
        """
            Storing the complete transactions of the database/input file in a database variable
        :return: None
        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            data, ts = [], []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'TS' in i:
                ts = self._iFile['TS'].tolist()
            if 'Transactions' in i:
                data = self._iFile['Transactions'].tolist()
            for i in range(len(data)):
                tr = [ts[i][0]]
                tr = tr + data[i]
                self._Database.append(tr)

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

    def _convert(self, value) -> int:
        """
        To convert the given user specified value

        :param value: user specified value
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

    @deprecated("It is recommended to use mine() instead of startMine() for mining process")
    def startMine(self) -> None:
        """
        Mining process will start from this function
        :return: None
        """

        self.Mine()

    def _getMaxPer(self, arr, maxTS):
        arr = np.append(arr, [0, maxTS])
        arr = np.sort(arr)
        arr = np.diff(arr)

        return np.max(arr)

    def _construct(self, items, data, minSup, maxPer, maxTS, patterns):

        # maxPerItems = {k: self.getMaxPer(v, maxTS) for k, v in items.items() if len(v) >= minSup}

        items = {k: v for k, v in items.items() if len(v) >= minSup and self._getMaxPer(v, maxTS) <= maxPer}

        #tested ok
        for item, ts in items.items():
            # pat = "\t".join(item)
            # self.patCount += 1
            # patterns[pat] = (len(ts), self.getMaxPer(ts, maxTS))
            patterns[tuple([item])] = [len(ts), self._getMaxPer(ts, maxTS)]

        root = _Node([], None, None)
        itemNodes = {}
        for line in data:
            currNode = root
            index = int(line[0])
            line = line[1:]
            line = sorted([item for item in line if item in items], key = lambda x: len(items[x]), reverse = True)
            for item in line:
                currNode = currNode.addChild(item, [index])   # heavy
                if item in itemNodes:
                    itemNodes[item].add(currNode)
                else:
                    itemNodes[item] = set([currNode])

        return root, itemNodes


    def _recursive(self, root, itemNode, minSup, maxPer, patterns, maxTS):

        for item in itemNode:
            newRoot = _Node(root.item + [item], None, None)

            itemLocs = {}
            transactions = {}
            for node in itemNode[item]:
                transaction, locs = node.traverse()
                if len(transaction) < 1:
                    continue
                # transactions.append((transaction, locs))
                if tuple(transaction) in transactions:
                    transactions[tuple(transaction)].extend(locs)
                else:
                    transactions[tuple(transaction)] = locs

                for item in transaction:
                    if item in itemLocs:
                        itemLocs[item] += locs
                    else:
                        itemLocs[item] = list(locs)

            # Precompute getMaxPer results for itemLocs
            maxPerResults = {item: self._getMaxPer(itemLocs[item], maxTS) for item in itemLocs if len(itemLocs[item]) >= minSup}

            # Filter itemLocs based on minSup and maxPer
            itemLocs = {k: len(v) for k, v in itemLocs.items() if k in maxPerResults and maxPerResults[k] <= maxPer}

            # Iterate over filtered itemLocs
            for item in itemLocs:
                # pat = "\t".join([str(x) for x in newRoot.item + [item]])
                # self.patCount += 1
                # patterns[pat] = [itemLocs[item], maxPerResults[item]]
                patterns[tuple(newRoot.item + [item])] = [itemLocs[item], maxPerResults[item]]
            
            if not itemLocs:
                continue

            newItemNodes = {}

            for transaction, locs in transactions.items():
                transaction = sorted([item for item in transaction if item in itemLocs], key = lambda x: itemLocs[x], reverse = True)
                if len(transaction) < 1:
                    continue
                currNode = newRoot
                for item in transaction:
                    currNode = currNode.addChild(item, locs)
                    if item in newItemNodes:
                        newItemNodes[item].add(currNode)
                    else:
                        newItemNodes[item] = set([currNode])

            self._recursive(newRoot, newItemNodes, minSup, maxPer, patterns, _lno)

    def Mine(self) -> None:
        """
        Mining process will start from this function
        :return: None
        """

        global _minSup, _maxPer, _lno
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._minSup is None:
            raise Exception("Please enter the Minimum Support")
        if self._maxPer is None:
            raise Exception("Please enter the Maximum Periodicity")
        if self._sep is None:
            raise Exception("Default separator is tab space, please enter the separator if you have different separator in the input file")

        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        self._maxPer = self._convert(self._maxPer)
        #tested ok
        _minSup, _maxPer, _lno = self._minSup, self._maxPer, len(self._Database)
        if self._minSup > len(self._Database):
            raise Exception("Please enter the minSup in range between 0 to 1")
        

        items = {}

        # tested ok
        for line in self._Database:
            index = int(line[0])
            for item in line[1:]:
                if item not in items:
                    items[item] = []
                items[item].append(index)

        root, itemNodes = self._construct(items, self._Database, _minSup, _maxPer, _lno, self._finalPatterns)

        self._recursive(root, itemNodes, _minSup, _maxPer, self._finalPatterns, _lno)

    

        newPattern = {}
        for k, v in self._finalPatterns.items():
            newPattern["\t".join([str(x) for x in k])] = v

        self._finalPatterns = newPattern




        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Periodic Frequent patterns were generated successfully using PFPGrowth algorithm ")

    def getMemoryUSS(self) -> float:
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self) -> float:
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self) -> float:
        """Calculating the total amount of runtime taken by the mining process


        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self) -> _ab._pd.DataFrame:
        """
        Storing final periodic-frequent patterns in a dataframe

        :return: returning periodic-frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b[0], b[1]])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
        return dataFrame

    def save(self, outFile: str) -> None:
        """
        Complete set of periodic-frequent patterns will be loaded in to an output file

        :param outFile: name of the output file
        :type outFile: csv file
        :return: None
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x + ":" + str(y[0]) + ":" + str(y[1])
            #s1 = x.replace(' ', '\t').strip() + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % s1)

    def getPatterns(self) -> Dict[str, Tuple[int, int]]:
        """
        Function to send the set of periodic-frequent patterns after completion of the mining process

        :return: returning periodic-frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self) -> None:
        """
        This function is used to print the results
        :return: None
        """
        print("Total number of Periodic Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = PFPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = PFPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        print("Total number of Periodic-Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")



    """
    **Methods to execute code on terminal**
    --------------------------------------------
    .. code-block:: console

      Format:
            
      (.venv) $ python3 PFPGrowth.py <inputFile> <outputFile> <minSup> <maxPer>

      Example:
      
      (.venv) $ python3 PFPGrowth.py sampleTDB.txt patterns.txt 0.3 0.4

    .. note:: minSup will be considered in percentage of database transactions

    **Importing this algorithm into a python program**
    ---------------------------------------------------
    .. code-block:: python

                from PAMI.periodicFrequentPattern.basic import PFPGrowth as alg

                obj = alg.PFPGrowth(iFile, minSup, maxPer)

                obj.startMine()

                periodicFrequentPatterns = obj.getPatterns()

                print("Total number of Periodic Frequent Patterns:", len(periodicFrequentPatterns))

                obj.save(oFile)

                Df = obj.getPatternsAsDataFrame()

                memUSS = obj.getMemoryUSS()

                print("Total Memory in USS:", memUSS)

                memRSS = obj.getMemoryRSS()

                print("Total Memory in RSS", memRSS)

                run = obj.getRuntime()

                print("Total ExecutionTime in seconds:", run)
    """