


# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#
#     from PAMI.periodicFrequentPattern.maximal import ThreePGrowth as alg
#
#     obj = alg.ThreePGrowth(iFile, periodicSupport, period)
#
#     obj.startMine()
#
#     partialPeriodicPatterns = obj.partialPeriodicPatterns()
#
#     print("Total number of partial periodic Patterns:", len(partialPeriodicPatterns))
#
#     obj.save(oFile)
#
#     Df = obj.getPatternInDf()
#
#     memUSS = obj.getMemoryUSS()
#
#     print("Total Memory in USS:", memUSS)
#
#     memRSS = obj.getMemoryRSS()
#
#     print("Total Memory in RSS", memRSS)
#
#     run = obj.getRuntime()
#
#     print("Total ExecutionTime in seconds:", run)

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

import sys as _sys
import validators as _validators
from urllib.request import urlopen as _urlopen
from PAMI.partialPeriodicPattern.maximal import abstract as _abstract

global maximalTree
_periodicSupport = float()
_period = float()
_lno = int()



class _Node(object):
    """
    A class used to represent the node of frequentPatternTree

    ...

    Attributes:
    ----------
        item : int
            storing item of a node
        timeStamps : list
            To maintain the timestamps of Database at the end of the branch
        parent : node
            To maintain the parent of every node
        children : list
            To maintain the children of node

    Methods:
    -------

        addChild(itemName)
            storing the children to their respective parent nodes
    """

    def __init__(self, item, children):
        self.item = item
        self.children = children
        self.parent = None
        self.timeStamps = []

    def _addChild(self, node):
        """
        To add the children details to the parent node children list

        :param node: children node

        :return: adding to parent node children
        """
        self.children[node.item] = node
        node.parent = self


class _Tree(object):
    """
    A class used to represent the frequentPatternGrowth tree structure

    ...

    Attributes:
    ----------
        root : Node
            Represents the root node of the tree
        summaries : dictionary
            storing the nodes with same item name
        info : dictionary
            stores the support of items


    Methods:
    -------
        addTransaction(Database)
            creating Database as a branch in frequentPatternTree
        getConditionPatterns(Node)
            generates the conditional patterns from tree for specific node
        conditionalTransaction(prefixPaths,Support)
            takes the prefixPath of a node and support at child of the path and extract the frequent items from
            prefixPaths and generates prefixPaths with items which are frequent
        remove(Node)
            removes the node from tree once after generating all the patterns respective to the node
        generatePatterns(Node)
            starts from the root node of the tree and mines the frequent patterns
    """

    def __init__(self):
        self.root = _Node(None, {})
        self.summaries = {}
        self.info = {}
        #self.maximalTree = _MPTree()

    def _addTransaction(self, transaction, tid):
        """
        adding transaction into database

        :param transaction: transactions in a database

        :param tid: timestamp of the transaction in database

        :return: pftree
        """
        currentNode = self.root
        for i in range(len(transaction)):
            if transaction[i] not in currentNode.children:
                newNode = _Node(transaction[i], {})
                currentNode._addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]
        currentNode.timeStamps = currentNode.timeStamps + tid

    def _getConditionalPatterns(self, alpha):
        """
        to get the conditional patterns of a node

        :param alpha: node in the tree

        :return: conditional patterns of a node
        """
        finalPatterns = []
        finalSets = []
        for i in self.summaries[alpha]:
            set1 = i.timeStamps
            set2 = []
            while i.parent.item is not None:
                set2.append(i.parent.item)
                i = i.parent
            if len(set2) > 0:
                set2.reverse()
                finalPatterns.append(set2)
                finalSets.append(set1)
        finalPatterns, finalSets, info = _conditionalTransactions(finalPatterns, finalSets)
        return finalPatterns, finalSets, info

    def _removeNode(self, nodeValue):
        """
        removes the leaf node by pushing its timestamps to parent node

        :param nodeValue: node of a tree

        :return:
        """
        for i in self.summaries[nodeValue]:
            i.parent.timeStamps = i.parent.timeStamps + i.timeStamps
            del i.parent.children[nodeValue]
            i = None

    def _getTimeStamps(self, alpha):
        """
        to get all the timestamps related to a node in tree

        :param alpha: node of a tree

        :return: timestamps of a node
        """
        temp = []
        for i in self.summaries[alpha]:
            temp += i.timeStamps
        return temp

    def _generatePatterns(self, prefix, _patterns, maximalTree):
        """
            To generate the maximal periodic frequent patterns

            :param prefix: an empty list of itemSet to form the combinations

            :return: maximal periodic frequent patterns
        """

        for i in sorted(self.summaries, key=lambda x: (self.info.get(x), -x)):
            pattern = prefix[:]
            pattern.append(i)
            condPattern, timeStamps, info = self._getConditionalPatterns(i)
            conditionalTree = _Tree()
            conditionalTree.info = info.copy()
            head = pattern[:]
            tail = []
            for k in info:
                tail.append(k)
            sub = head + tail
            if maximalTree._checkerSub(sub) == 1:
                for pat in range(len(condPattern)):
                    conditionalTree._addTransaction(condPattern[pat], timeStamps[pat])
                if len(condPattern) >= 1:
                    conditionalTree._generatePatterns(pattern, _patterns, maximalTree)
                else:
                    maximalTree._addTransaction(pattern)
                    _patterns[tuple(pattern)] = self.info[i]
            self._removeNode(i)


class _MNode(object):
    """
    A class used to represent the node of frequentPatternTree

    ...

    Attributes:
    ----------
        item : int
            storing item of a node
        children : list
            To maintain the children of node

    Methods:
    -------

        addChild(itemName)
            storing the children to their respective parent nodes
    """

    def __init__(self, item, children):
        self.item = item
        self.children = children

    def _addChild(self, node):
        """
        To add the children details to parent node children variable

        :param node: children node

        :return: adding children node to parent node
        """
        self.children[node.item] = node
        node.parent = self


class _MPTree(object):
    """
    A class used to represent the node of frequentPatternTree

    ...

    Attributes:
    ----------
        root : node
            the root of a tree
        summaries : dict
            to store the items with same name into dictionary

    Methods:
    -------
        checkerSub(itemSet)
            to check of subset of itemSet is present in tree
    """

    def __init__(self):
        self.root = _Node(None, {})
        self.summaries = {}

    def _addTransaction(self, transaction):
        """
        to add the transaction in maximal tree

        :param transaction: resultant periodic frequent pattern

        :return: maximal tree
        """
        currentNode = self.root
        transaction.sort()
        for i in range(len(transaction)):
            if transaction[i] not in currentNode.children:
                newNode = _MNode(transaction[i], {})
                currentNode._addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].insert(0, newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]

    def _checkerSub(self, items):
        """
        To check subset present of items in the maximal tree

        :param items: the pattern to check for subsets

        :return: 1
        """
        items.sort(reverse=True)
        item = items[0]
        if item not in self.summaries:
            return 1
        else:
            if len(items) == 1:
                return 0
        for t in self.summaries[item]:
            cur = t.parent
            i = 1
            while cur.item is not None:
                if items[i] == cur.item:
                    i += 1
                    if i == len(items):
                        return 0
                cur = cur.parent
        return 1


#maximalTree = _MPTree()


def _getPeriodAndSupport(timeStamps):
    """
    To calculate the periodicity and support of a pattern with their respective timeStamps

    :param timeStamps: timeStamps

    :return: Support and periodicity
    """
    timeStamps.sort()
    per = 0
    for i in range(len(timeStamps) - 1):
        j = i + 1
        if abs(timeStamps[j] - timeStamps[i]) <= _period:
            per += 1
    return per


def _conditionalTransactions(condPatterns, condTimeStamps):
    """
    To calculate the timestamps of conditional items in conditional patterns

    :param condPatterns: conditional patterns of node

    :param condTimeStamps: timeStamps of a conditional patterns

    :return: removing items with low periodicSupport or periodicity and sort the conditional transactions
    """
    pat = []
    timeStamps = []
    data1 = {}
    for i in range(len(condPatterns)):
        for j in condPatterns[i]:
            if j in data1:
                data1[j] = data1[j] + condTimeStamps[i]
            else:
                data1[j] = condTimeStamps[i]
    updatedDict = {}
    for m in data1:
        updatedDict[m] = _getPeriodAndSupport(data1[m])
    updatedDict = {k: v for k, v in updatedDict.items() if v >= _periodicSupport}
    count = 0
    for p in condPatterns:
        p1 = [v for v in p if v in updatedDict]
        trans = sorted(p1, key=lambda x: (updatedDict.get(x), -x), reverse=True)
        if len(trans) > 0:
            pat.append(trans)
            timeStamps.append(condTimeStamps[count])
        count += 1
    return pat, timeStamps, updatedDict


class Max3PGrowth(_abstract._partialPeriodicPatterns):
    """
    Description:
    ------------
        Max3p-Growth algorithm IS to discover maximal periodic-frequent patterns in a temporal database.
        It extract the partial periodic patterns from 3p-tree and checks for the maximal property and stores
        all the maximal patterns in max3p-tree and extracts the maximal periodic patterns.

    Reference:
    -----------
        R. Uday Kiran, Yutaka Watanobe, Bhaskar Chaudhury, Koji Zettsu, Masashi Toyoda, Masaru Kitsuregawa,
        "Discovering Maximal Periodic-Frequent Patterns in Very Large Temporal Databases",
        IEEE 2020, https://ieeexplore.ieee.org/document/9260063

    Attributes:
    -----------

        iFile : file
            Name of the Input file or path of the input file
        oFile : file
            Name of the output file or path of the output file
        periodicSupport: float or int or str
            The user can specify periodicSupport either in count or proportion of database size.
            If the program detects the data type of periodicSupport is integer, then it treats periodicSupport is expressed in count.
            Otherwise, it will be treated as float.
            Example: periodicSupport=10 will be treated as integer, while periodicSupport=10.0 will be treated as float
        period: float or int or str
            The user can specify period either in count or proportion of database size.
            If the program detects the data type of period is integer, then it treats period is expressed in count.
            Otherwise, it will be treated as float.
            Example: period=10 will be treated as integer, while period=10.0 will be treated as float
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
        periodicSupport : int/float
            The user given minimum support
        period : int/float
            The user given maximum period
        Database : list
            To store the transactions of a database in list
        mapSupport : Dictionary
            To maintain the information of item and their frequency
        lno : int
            it represents the total no of transaction
        tree : class
            it represents the Tree class
        itemSetCount : int
            it represents the total no of patterns
        finalPatterns : dict
            it represents to store the patterns

    Methods:
    ---------
        startMine()
            Mining process will start from here
        getFrequentPatterns()
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
        creatingitemSets(fileName)
            Scans the dataset or dataframes and stores in list format
        PeriodicFrequentOneItem()
            Extracts the one-periodic-frequent patterns from Databases
        updateDatabases()
            update the Databases by removing aperiodic items and sort the Database by item decreased support
        buildTree()
            after updating the Databases ar added into the tree by setting root node as null
        startMine()
            the main method to run the program

    Executing the code on terminal:
    -------------------------------
        Format:
        --------
            >>> python3 max3prowth.py <inputFile> <outputFile> <periodicSupport> <period>

        Examples:
        ---------
            >>>  python3 Max3PGrowth.py sampleTDB.txt patterns.txt 0.3 0.4  (periodicSupport will be considered in percentage of database
                transactions)

            >>>  python3 Max3PGrowth.py sampleTDB.txt patterns.txt 3 4  (periodicSupport will be considered in count)

    Sample run of the importing code:
    ----------------------------------
    .. code-block:: python

            from PAMI.periodicFrequentPattern.maximal import ThreePGrowth as alg

            obj = alg.ThreePGrowth(iFile, periodicSupport, period)

            obj.startMine()

            partialPeriodicPatterns = obj.partialPeriodicPatterns()

            print("Total number of partial periodic Patterns:", len(partialPeriodicPatterns))

            obj.save(oFile)

            Df = obj.getPatternInDf()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


    Credits:
    ---------
        The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.\n

    """
    _startTime = float()
    _endTime = float()
    _periodicSupport = str()
    _period = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _rank = {}
    _rankDup = {}
    _lno = 0
    _patterns = {}
    _pfList = {}
    _maximalTree = str()

    def _creatingitemSets(self):
        """ Storing the complete Databases of the database/input file in a database variable
            :rtype: storing transactions into Database variable
        """

        self._Database = []
        if isinstance(self._iFile, _abstract._pd.DataFrame):
            timeStamp, data = [], []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'TS' in i:
                timeStamp = self._iFile['TS'].tolist()
            if 'Transactions' in i:
                data = self._iFile['Transactions'].tolist()
            for i in range(len(data)):
                tr = [timeStamp[i]]
                tr = tr + data[i]
                self._Database.append(tr)
            self._lno = len(self._Database)
            # print(self.Database)
        if isinstance(self._iFile, str):
            if _validators.url(self._iFile):
                data = _urlopen(self._iFile)
                for line in data:
                    self._lno += 1
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            self._lno += 1
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def _periodicFrequentOneItem(self):
        """
            calculates the support of each item in the dataset and assign the ranks to the items
            by decreasing support and returns the frequent items list
            :rtype: return the one-length periodic frequent patterns


            """
        self._pfList = {}
        data = {}
        for tr in self._Database:
            for i in range(1, len(tr)):
                if tr[i] not in data:
                    data[tr[i]] = [0, int(tr[0]), 1]
                else:
                    lp = abs(int(tr[0]) - data[tr[i]][1])
                    if lp <= _period:
                        data[tr[i]][0] += 1
                    data[tr[i]][1] = int(tr[0])
                    data[tr[i]][2] += 1
        data = {k: v[0] for k, v in data.items() if v[0] >= self._periodicSupport}
        self._pfList = [k for k, v in sorted(data.items(), key=lambda x: x[1], reverse=True)]
        self._rank = dict([(index, item) for (item, index) in enumerate(self._pfList)])
        return data

    def _updateDatabases(self, dict1):
        """ Remove the items which are not frequent from Databases and updates the Databases with rank of items

            :param dict1: frequent items with support
            :type dict1: dictionary
            :rtype: sorted and updated transactions
            """
        list1 = []
        for tr in self._Database:
            list2 = [int(tr[0])]
            for i in range(1, len(tr)):
                if tr[i] in dict1:
                    list2.append(self._rank[tr[i]])
            if len(list2) >= 2:
                basket = list2[1:]
                basket.sort()
                list2[1:] = basket[0:]
                list1.append(list2)
        return list1

    @staticmethod
    def _buildTree(data, info):
        """ it takes the Databases and support of each item and construct the main tree with setting root node as null

            :param data: it represents the one Databases in database
            :type data: list
            :param info: it represents the support of each item
            :type info: dictionary
            :rtype: returns root node of tree
        """

        rootNode = _Tree()
        rootNode.info = info.copy()
        for i in range(len(data)):
            set1 = [data[i][0]]
            rootNode._addTransaction(data[i][1:], set1)
        return rootNode

    def _convert(self, value):
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

    def _convertItems(self, itemSet):
        """

        to convert the maximal pattern items with their original item names

        :param itemSet: maximal periodic frequent pattern

        :return: pattern with original item names
        """
        t1 = []
        for i in itemSet:
            t1.append(self._pfList[i])
        return t1

    def startMine(self):
        """ Mining process will start from this function
        """

        global _periodicSupport, _period, _lno
        self._startTime = _abstract._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._periodicSupport is None:
            raise Exception("Please enter the Minimum Support")
        self._creatingitemSets()
        self._periodicSupport = self._convert(self._periodicSupport)
        self._period = self._convert(self._period)
        _periodicSupport, _period, _lno = self._periodicSupport, self._period, len(self._Database)
        if self._periodicSupport > len(self._Database):
            raise Exception("Please enter the periodicSupport in range between 0 to 1")
        generatedItems = self._periodicFrequentOneItem()
        updatedDatabases = self._updateDatabases(generatedItems)
        for x, y in self._rank.items():
            self._rankDup[y] = x
        info = {self._rank[k]: v for k, v in generatedItems.items()}
        Tree = self._buildTree(updatedDatabases, info)
        self._patterns = {}
        self._maximalTree = _MPTree()
        Tree._generatePatterns([], self._patterns, self._maximalTree)
        self._finalPatterns = {}
        for x, y in self._patterns.items():
            st = str()
            x = self._convertItems(x)
            for k in x:
                st = st + k + "\t"
            self._finalPatterns[st] = y
        self._endTime = _abstract._time.time()
        process = _abstract._psutil.Process(_abstract._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Maximal Partial Periodic Frequent patterns were generated successfully using MAX-3PGrowth algorithm ")

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
        """Storing final periodic-frequent patterns in a dataframe

        :return: returning periodic-frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a.replace('\t', ' '), b])
            dataFrame = _abstract._pd.DataFrame(data, columns=['Patterns', 'periodicSupport'])
        return dataFrame

    def save(self, outFile):
        """Complete set of periodic-frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of periodic-frequent patterns after completion of the mining process

        :return: returning periodic-frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        print("Total number of  Maximal Partial Periodic Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_sys.argv) == 5 or len(_sys.argv) == 6:
        if len(_sys.argv) == 6:
            _ap = Max3PGrowth(_sys.argv[1], _sys.argv[3], _sys.argv[4], _sys.argv[5])
        if len(_sys.argv) == 5:
            _ap = Max3PGrowth(_sys.argv[1], _sys.argv[3], _sys.argv[4])
        _ap.startMine()
        print("Total number of Maximal Partial Periodic Patterns:", len(_ap.getPatterns()))
        _ap.save(_sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        for i in [100, 200, 300, 400, 500]:
            _ap = Max3PGrowth('/Users/Likhitha/Downloads/temporal_T10I4D100K.csv', i, 5000, '\t')
            _ap.startMine()
            print("Total number of Maximal Partial Periodic Patterns:", len(_ap.getPatterns()))
            _ap.save('/Users/Likhitha/Downloads/output.txt')
            print("Total Memory in USS:", _ap.getMemoryUSS())
            print("Total Memory in RSS", _ap.getMemoryRSS())
            print("Total ExecutionTime in ms:", _ap.getRuntime())
        print("Error! The number of input parameters do not match the total number of parameters provided")
