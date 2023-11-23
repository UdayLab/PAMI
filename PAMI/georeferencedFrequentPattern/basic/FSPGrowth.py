# FSPGrowth is a transactional database and a spatial (or neighborhood) file, FSPM aims to discover all of those patterns
# that satisfy the user-specified minimum support (minSup) and maximum distance (maxDist) constraints
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#         from PAMI.georeferencedFrequentPattern.basic import FSPGrowth as alg
#
#         obj = alg.FSPGrowth("sampleTDB.txt", "sampleN.txt", 5)
#
#         obj.startMine()
#
#         spatialFrequentPatterns = obj.getPatterns()
#
#         print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))
#
#         obj.save("outFile")
#
#         memUSS = obj.getMemoryUSS()
#
#         print("Total Memory in USS:", memUSS)
#
#         memRSS = obj.getMemoryRSS()
#
#         print("Total Memory in RSS", memRSS)
#
#         run = obj.getRuntime()
#
#         print("Total ExecutionTime in seconds:", run)
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


from PAMI.georeferencedFrequentPattern.basic import abstract as _ab
from typing import List, Dict, Tuple, Set, Union, Any, Generator


class _Node:
    """
    A class used to represent the node of frequentPatternTree
    Attributes
    ----------
        item : int
            Storing item of a node
        count : int
            To maintain the support of node
        children : dict
            To maintain the children of node
        prefix : list
            To maintain the prefix of node
    """

    def __init__(self, item, count, children):
        self.item = item
        self.count = count
        self.children = children
        self.prefix = []


class _Tree:
    """
    A class used to represent the frequentPatternGrowth tree structure
    Attributes
    ----------
        root : Node
            The first node of the tree set to Null.
        nodeLink : dict
            Stores the nodes which shares same item
    Methods
    -------
        createTree(transaction,count)
            Adding transaction into the tree
        linkNode(node)
            Adding node to nodeLink
        createCPB(item,neighbour)
            Create conditional pattern base based on item and neighbour
        mergeTree(tree,fpList)
            Merge tree into yourself
        createTransactions(fpList)
            Create transactions from yourself
        getPattern(item,suffixItem,minSup,neighbour)
            Get frequent patterns based on suffixItem
        mining(minSup,isResponsible = lambda x:True,neighbourhood=None)
            Mining yourself
    """

    def __init__(self):
        self.root = _Node(None, 0, {})
        self.nodeLink = _ab._OrderedDict()

    def createTree(self, transaction, count):
        """
        Create tree or add transaction into yourself.
        :param transaction: list
        :param count: int
        :return: Tree
        """
        current = self.root
        for item in transaction:
            if item not in current.children:
                current.children[item] = _Node(item, count, {})
                current.children[item].prefix = transaction[0:transaction.index(item)]
                self.linkNode(current.children[item])
            else:
                current.children[item].count += count
            current = current.children[item]
        return self

    def linkNode(self, node):
        """
        Maintain link of node by adding node to nodeLink
        :param node: Node
        :return:
        """
        if node.item in self.nodeLink:
            self.nodeLink[node.item].append(node)
        else:
            self.nodeLink[node.item] = []
            self.nodeLink[node.item].append(node)

    def createCPB(self, item, neighbour) :
        """
        Create conditional pattern base based on item and neighbour
        :param item: int
        :param neighbour: dict
        :return: Tree
        """
        pTree = _Tree()
        for node in self.nodeLink[item]:
            # print(node.item, neighbour[node.item])
            if node.item in neighbour:
                node.prefix = [item for item in node.prefix if item in neighbour.get(node.item)]
            pTree.createTree(node.prefix, node.count)
        return pTree

    def mergeTree(self, tree, fpList):
        """
        Merge tree into yourself
        :param tree: Tree
        :param fpList: list
        :return: Tree
        """
        transactions = tree.createTransactions(fpList)
        for transaction in transactions:
            self.createTree(transaction, 1)
        return self

    def createTransactions(self, fpList):
        """
        Create transactions that configure yourself
        :param fpList: list
        :return: list
        """
        transactions = []
        flist = [x for x in fpList if x in self.nodeLink]
        for item in reversed(flist):
            for node in self.nodeLink[item]:
                if node.count != 0:
                    transaction = node.prefix
                    transaction.append(node.item)
                    transactions.extend([transaction for i in range(node.count)])
                    current = self.root
                    for i in transaction:
                        current = current.children[i]
                        current.count -= node.count
        return transactions

    def getPattern(self, item, suffixItem, minSup, neighbour):
        """
        Get frequent patterns based on suffixItem
        :param item: int
        :param suffixItem: tuple
        :param minSup: int
        :param neighbour: dict
        :return: list
        """
        pTree = self.createCPB(item, neighbour)
        frequentItems = {}
        frequentPatterns = []
        for i in pTree.nodeLink.keys():
            frequentItems[i] = 0
            for node in pTree.nodeLink[i]:
                frequentItems[i] += node.count
        frequentItems = {key: value for key, value in frequentItems.items() if value >= minSup}
        for i in frequentItems:
            pattern = suffixItem + "\t" + i
            frequentPatterns.append((pattern, frequentItems[i]))
            frequentPatterns.extend(pTree.getPattern(i, pattern, minSup, neighbour))
        return frequentPatterns

    def mining(self, minSup, neighbourhood: [Dict[int, List[int]]] = None):
        """
        Pattern mining on your own
        :param minSup: int
        :param neighbourhood: function
        :param neighbourhood: dict
        :return: list
        """
        frequentPatterns = []
        flist = sorted([item for item in self.nodeLink.keys()])
        for item in reversed(flist):
            frequentPatterns.extend(self.getPattern(item, item, minSup, neighbourhood))
        return frequentPatterns


class FSPGrowth(_ab._spatialFrequentPatterns):
    """
    Description:
    -------------
        Given a transactional database and a spatial (or neighborhood) file, FSPM aims to discover all of those patterns
        that satisfy the user-specified minimum support (minSup) and maximum distance (maxDist) constraints
    Reference:
    -----------
        Rage, Uday & Fournier Viger, Philippe & Zettsu, Koji & Toyoda, Masashi & Kitsuregawa, Masaru. (2020).
        Discovering Frequent Spatial Patterns in Very Large Spatiotemporal Databases.

    Attributes:
    ------------
        iFile : file
            Input file name or path of the input file
        nFile : file
            Neighbourhood file name or path of the neighbourhood file
        oFile : file
            Name of the output file or the path of output file
        minSup : float
            The user can specify minSup either in count or proportion of database size.
        finalPatterns : dict
            Storing the complete set of patterns in a dictionary variable
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program

    Methods:
    --------
        startMine()
            This function starts pattern mining.
        getPatterns()
            Complete set of patterns will be retrieved with this function
        save(oFile)
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsInDataFrame()
            Complete set of frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        getNeighbour(string)
            This function changes string to tuple(neighbourhood).
        getFrequentItems(database)
            This function create frequent items from database.
        genCondTransaction(transaction, rank)
            This function generates conditional transaction for processing on each workers.
        getPartitionId(item)
            This function generates partition id
        mapNeighbourToNumber(neighbour, rank)
            This function maps neighbourhood to number.
            Because in this program, each item is mapped to number based on fpList so that it can be distributed.
            So the contents of neighbourhood must also be mapped to a number.
        createFPTree()
            This function creates FPTree.
        getAllFrequentPatterns(data, fpList, ndata)
            This function generates all frequent patterns

    Executing the code on terminal :
    ---------------------------------
        Format:

            >>> python3 FSPGrowth.py <inputFile> <outputFile> <neighbourFile> <minSup>
        Examples:

            >>> python3 FSPGrowth.py sampleTDB.txt output.txt sampleN.txt 0.5 (minSup will be considered in percentage of database transactions)

    Sample run of importing the code :
    -----------------------------------
    .. code-block:: python

        from PAMI.georeferencedFrequentPattern.basic import FSPGrowth as alg

        obj = alg.FSPGrowth("sampleTDB.txt", "sampleN.txt", 5)

        obj.startMine()

        spatialFrequentPatterns = obj.getPatterns()

        print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))

        obj.save("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    Credits:
    ----------
        The complete program was written by Yudai Masu under the supervision of Professor Rage Uday Kiran.
    """

    _minSup = float()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _nFile = " "
    _oFile = " "
    _sep = " "
    _lno = 0
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _neighbourList = {}
    _fpList = []

    def _readDatabase(self):
        """
        Read input file and neighborhood file
        """

        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
            self._lno = len(self._Database)
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    self._lno += 1
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            self._lno += 1
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(temp)
                except IOError:
                    print("File Not Found1")
                    quit()

        self._neighbourList = {}
        if isinstance(self._nFile, _ab._pd.DataFrame):
            data, items = [], []
            if self._nFile.empty:
                print("its empty..")
            i = self._nFile.columns.values.tolist()
            if 'item' in i:
                items = self._nFile['items'].tolist()
            if 'Neighbours' in i:
                data = self._nFile['Neighbours'].tolist()
            for k in range(len(items)):
                self._neighbourList[items[k][0]] = data[k]
            # print(self.Database)
        if isinstance(self._nFile, str):
            if _ab._validators.url(self._nFile):
                data = _ab._urlopen(self._nFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._neighbourList[temp[0]] = temp[1:]
            else:
                try:
                    with open(self._nFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._neighbourList[temp[0]] = temp[1:]
                except IOError:
                    print("File Not Found2")
                    quit()

    def _getFrequentItems(self):
        """
        Create frequent items and self.fpList from self.Database
        """
        oneFrequentItem = {}
        for transaction in self._Database:
            for item in transaction:
                oneFrequentItem[item] = oneFrequentItem.get(item, 0) + 1
        self._finalPatterns = {key: value for key, value in oneFrequentItem.items() if value >= self._minSup}
        self._fpList = list(dict(sorted(oneFrequentItem.items(), key=lambda x: x[1], reverse=True)))

    def _createFPTree(self):
        """ create FP Tree and self.fpList from self.Database"""
        FPTree = _Tree()
        for transaction in self._Database:
            FPTree.createTree(transaction, 1)
        return FPTree

    def _sortTransaction(self):
        """
        Sort each transaction of self.Database based on self.fpList
        """
        for i in range(len(self._Database)):
            self._Database[i] = [item for item in self._Database[i] if item in self._fpList]
            self._Database[i].sort(key=lambda value: self._fpList.index(value))

    def _convert(self, value):
        """
        To convert the given user specified value
        :param value: user specified value
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self._lno * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self._lno * value)
            else:
                value = int(value)
        return value

    def startMine(self):
        """
        start pattern mining from here
        """
        self._startTime = _ab._time.time()
        self._finalPatterns = {}
        self._readDatabase()
        print(len(self._Database), len(self._neighbourList))
        self._minSup = self._convert(self._minSup)
        self._getFrequentItems()
        self._sortTransaction()
        _FPTree = self._createFPTree()
        self._finalPatterns.update(dict(_FPTree.mining(self._minSup, self._neighbourList)))
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent Spatial Patterns successfully generated using FSPGrowth")

    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function
        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function
        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process
        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe
        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataframe = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a.replace('\t', ' '), b])
            dataframe = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataframe

    def save(self, oFile):
        """
        Complete set of frequent patterns will be loaded in to a output file
        :param oFile: name of the output file
        :type oFile: file
        """
        self._oFile = oFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """

        return self._finalPatterns

    def printResults(self):
        """ This function is used to print the results
        """
        print("Total number of Spatial Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = FSPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = FSPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        print("Total number of Spatial Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in seconds:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")

