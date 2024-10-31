#  PS-Growth is one of the fundamental algorithm to discover periodic-frequent patterns in a temporal database.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#
#             from PAMI.periodicFrequentPattern.basic import PSGrowth as alg
#
#             obj = alg.PSGrowth("../basic/sampleTDB.txt", "2", "6")
#
#             obj.startMine()
#
#             periodicFrequentPatterns = obj.getPatterns()
#
#             print("Total number of  Patterns:", len(periodicFrequentPatterns))
#
#             obj.save("patterns")
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
import pandas as pd
from deprecated import deprecated
from itertools import combinations as _combinations
from PAMI.periodicFrequentPattern.basic import abstract as _ab
from typing import List, Dict, Tuple, Set, Union, Any, Generator  

_pfList = []
_minSup = int()
_maxPer = int()
_lno = int()


class _Interval(object):
    """
    To represent the timestamp interval of a node in summaries
    """

    def __init__(self, start, end, per, sup) -> None:
        self.start = start
        self.end = end
        self.per = per
        self.sup = sup


class _NodeSummaries(object):
    """
    To define the summaries of timeStamps of a node

    :Attributes:

        totalSummaries : list
            stores the summaries of timestamps

    :Methods:

        insert(timeStamps)
            inserting and merging the timestamps into the summaries of a node
    """

    def __init__(self) -> None:
        self.totalSummaries = []

    def insert(self, tid) -> List[_Interval]:
        """ To insert and merge the timeStamps into summaries of a node
            :param tid: timeStamps of a node
            :return: summaries of a node
        """
        k = self.totalSummaries[-1]
        diff = tid - k.end
        if diff <= _maxPer:
            k.end = tid
            k.per = max(diff, k.per)
            #             print(k.per)
            k.sup += 1
        else:
            self.totalSummaries.append(_Interval(tid, tid, 0, 1))
        return self.totalSummaries


def _merge(summariesX, summariesY) -> List[_Interval]:
    """
    To Merge the timeStamps

    :param summariesX:  TimeStamps of a one itemSet
    :param summariesY:  TimeStamps of a one itemSet
    :return:  Merged timestamp of both itemSets
    """
    iter1 = 0
    iter2 = 0
    updatedSummaries = []
    l1 = len(summariesX)
    l2 = len(summariesY)
    while 1:
        if summariesX[iter1].start < summariesY[iter2].start:
            if summariesX[iter1].end < summariesY[iter2].start:
                diff = summariesY[iter2].start - summariesX[iter1].end
                if diff > _maxPer:
                    updatedSummaries.append(_Interval(summariesX[iter1].start,
                                                     summariesX[iter1].end, summariesX[iter1].per,
                                                     summariesX[iter1].sup))
                    iter1 += 1
                    if iter1 >= l1:
                        ck = 1
                        break
                else:
                    per1 = max(diff, summariesX[iter1].per)
                    per1 = max(per1, summariesY[iter2].per)
                    updatedSummaries.append(
                        _Interval(summariesX[iter1].start, summariesY[iter2].end, per1,
                                 summariesX[iter1].sup + summariesY[iter2].sup))
                    iter1 += 1
                    iter2 += 1
                    if iter1 >= l1:
                        ck = 1
                        break

                    if iter2 >= l2:
                        ck = 2
                        break

            else:
                if summariesX[iter1].end > summariesY[iter2].end:
                    updatedSummaries.append(_Interval(summariesX[iter1].start, summariesX[iter1].end,
                                                     summariesX[iter1].per,
                                                     summariesX[iter1].sup + summariesY[iter2].sup))
                else:
                    per1 = max(summariesX[iter1].per, summariesY[iter2].per)
                    updatedSummaries.append(
                        _Interval(summariesX[iter1].start, summariesY[iter2].end, per1,
                                 summariesX[iter1].sup + summariesY[iter2].sup))
                iter1 += 1
                iter2 += 1
                if iter1 >= l1:
                    ck = 1
                    break

                if iter2 >= l2:
                    ck = 2
                    break
        else:
            if summariesY[iter2].end < summariesX[iter1].start:
                diff = summariesX[iter1].start - summariesY[iter2].end
                if diff > _maxPer:
                    updatedSummaries.append(_Interval(summariesY[iter2].start, summariesY[iter2].end,
                                                     summariesY[iter2].per, summariesY[iter2].sup))
                    iter2 += 1
                    if iter2 >= l2:
                        ck = 2
                        break
                else:
                    per1 = max(diff, summariesY[iter2].per)
                    per1 = max(per1, summariesX[iter1].per)
                    updatedSummaries.append(
                        _Interval(summariesY[iter2].start, summariesX[iter1].end, per1,
                                 summariesY[iter2].sup + summariesX[iter1].sup))
                    iter2 += 1
                    iter1 += 1
                    if iter2 >= l2:
                        ck = 2
                        break

                    if iter1 >= l1:
                        ck = 1
                        break

            else:
                if summariesY[iter2].end > summariesX[iter1].end:
                    updatedSummaries.append(_Interval(summariesY[iter2].start, summariesY[iter2].end,
                                                     summariesY[iter2].per,
                                                     summariesY[iter2].sup + summariesX[iter1].sup))
                else:
                    per1 = max(summariesY[iter2].per, summariesX[iter1].per)
                    updatedSummaries.append(
                        _Interval(summariesY[iter2].start, summariesX[iter1].end, per1,
                                 summariesY[iter2].sup + summariesX[iter1].sup))
                iter2 += 1
                iter1 += 1
                if iter2 >= l2:
                    ck = 2
                    break

                if iter1 >= l1:
                    ck = 1
                    break
    if ck == 1:
        while iter2 < l2:
            updatedSummaries.append(summariesY[iter2])
            iter2 += 1
    else:
        while iter1 < l1:
            updatedSummaries.append(summariesX[iter1])
            iter1 += 1
    updatedSummaries = _update(updatedSummaries)

    return updatedSummaries


def _update(updatedSummaries) -> List[_Interval]:
    """ After updating the summaries with first, last, and period elements in summaries

    :param updatedSummaries: summaries that have been merged
    :return: updated summaries of a node
    """
    summaries = [updatedSummaries[0]]
    cur = updatedSummaries[0]
    for i in range(1, len(updatedSummaries)):
        v = (updatedSummaries[i].start - cur.end)
        if cur.end > updatedSummaries[i].start or v <= _maxPer:
            cur.end = max(updatedSummaries[i].end, cur.end)
            cur.sup += updatedSummaries[i].sup
            cur.per = max(cur.per, updatedSummaries[i].per)
            cur.per = max(cur.per, v)
        else:
            summaries.append(updatedSummaries[i])
        cur = summaries[-1]
    return summaries


class Node(object):
    """
    A class used to represent the node of frequentPatternTree

    :Attributes:

        item : int
            storing item of a node
        timeStamps : list
            To maintain the timeStamps of Database at the end of the branch
        parent : node
            To maintain the parent of every node
        children : list
            To maintain the children of node

    :Methods:

            addChild(itemName)
                storing the children to their respective parent nodes
        """

    def __init__(self, item, children) -> None:
        """
        Initializing the Node class

        :param item: Storing the item of a node
        :type item: int
        :param children: To maintain the children of a node
        :type children: dict
        :return: None
        """
        self.item = item
        self.children = children
        self.parent = None
        self.timeStamps = _NodeSummaries()

    def addChild(self, node) -> None:
        """
        Appends the children node details to a parent node

        :param node: children node
        :return: appending children node to parent node
        """
        self.children[node.item] = node
        node.parent = self


class _Tree(object):
    """

    A class used to represent the frequentPatternGrowth tree structure

    :Attributes:

        root : Node or None
            Represents the root node of the tree
        summaries : dictionary
            storing the nodes with same item name
        info : dictionary
            stores the support of items

    :Methods:

            addTransaction(Database)
                creating Database as a branch in frequentPatternTree
            addConditionalTransactions(prefixPaths, supportOfItems)
                construct the conditional tree for prefix paths
            getConditionalPatterns(Node)
                generates the conditional patterns from tree for specific node
            conditionalTransaction(prefixPaths,Support)
                takes the prefixPath of a node and support at child of the path and extract the frequent items from
                prefixPaths and generates prefixPaths with items which are frequent
            remove(Node)
                removes the node from tree once after generating all the patterns respective to the node
            generatePatterns(Node)
                starts from the root node of the tree and mines the periodic-frequent patterns

    """

    def __init__(self) -> None:
        self.root = Node(None, {})
        self.summaries = {}
        self.info = {}

    def addTransaction(self, transaction, tid) -> None:
        """
        Adding transaction into the tree

        :param transaction: it represents the one transaction in a database
        :type transaction: list
        :param tid: represents the timestamp of a transaction
        :type tid: list
        :return: None
        """
        currentNode = self.root
        for i in range(len(transaction)):
            if transaction[i] not in currentNode.children:
                newNode = Node(transaction[i], {})
                currentNode.addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]
        if len(currentNode.timeStamps.totalSummaries) != 0:
            currentNode.timeStamps.insert(tid)
        else:
            currentNode.timeStamps.totalSummaries.append(_Interval(tid, tid, 0, 1))

    def addConditionalPatterns(self, transaction, tid) -> None:
        """
        To add the conditional transactions in to conditional tree

        :param transaction: conditional transaction list of a node
        :param tid: timestamp of a conditional transaction
        :return: the conditional tree of a node
        """
        currentNode = self.root
        for i in range(len(transaction)):
            if transaction[i] not in currentNode.children:
                newNode = Node(transaction[i], {})
                currentNode.addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]
        if len(currentNode.timeStamps.totalSummaries) != 0:
            currentNode.timeStamps.totalSummaries = _merge(currentNode.timeStamps.totalSummaries, tid)
        else:
            currentNode.timeStamps.totalSummaries = tid

    def getConditionalPatterns(self, alpha) -> Tuple[List[List[int]], List[List[_Interval]], Dict[int, Tuple[int, int]]]:
        """
        To mine the conditional patterns of a node

        :param alpha: starts from the leaf node of a tree
        :return: the conditional patterns of a node
        """
        finalPatterns = []
        finalSets = []
        for i in self.summaries[alpha]:
            set1 = i.timeStamps.totalSummaries
            set2 = []
            while i.parent.item is not None:
                set2.append(i.parent.item)
                i = i.parent
            if len(set2) > 0:
                set2.reverse()
                finalPatterns.append(set2)
                finalSets.append(set1)
        finalPatterns, finalSets, info = conditionalTransactions(finalPatterns, finalSets)
        return finalPatterns, finalSets, info

    def removeNode(self, nodeValue) -> None:
        """
        to remove the node from the tree by pushing the timeStamps of leaf node to the parent node

        :param nodeValue: name of node to be deleted
        :return: removes the node from the tree
        """
        for i in self.summaries[nodeValue]:
            if len(i.parent.timeStamps.totalSummaries) != 0:
                i.parent.timeStamps.totalSummaries = _merge(i.parent.timeStamps.totalSummaries,
                                                           i.timeStamps.totalSummaries)
            else:
                i.parent.timeStamps.totalSummaries = i.timeStamps.totalSummaries
            del i.parent.children[nodeValue]
            del i
        del self.summaries[nodeValue]

    def getTimeStamps(self, alpha) -> List[_Interval]:
        """
        To get the timeStamps of a respective node

        :param alpha: name of node for the timeStamp
        :return: timeStamps of a node
        """
        temp = []
        for i in self.summaries[alpha]:
            temp += i.timeStamps
        return temp

    def check(self) -> int:
        """
        To the total number of child and their summaries
        :return: int
        """
        k = self.root
        while len(k.children) != 0:
            if len(k.children) > 1:
                return 1
            if len(k.children) != 0 and len(k.timeStamps.totalSummaries) > 0:
                return 1
            for j in k.children:
                v = k.children[j]
                k = v
        return -1

    def generatePatterns(self, prefix):
        """
        Generating the patterns from the tree

        :param prefix: empty list to form the combinations
        :return: returning the periodic-frequent patterns from the tree
        """
        global _pfList
        for i in sorted(self.summaries, key=lambda x: (self.info.get(x)[0], -x)):
            pattern = prefix[:]
            pattern.append(_pfList[i])
            yield pattern, self.info[i]
            patterns, timeStamps, info = self.getConditionalPatterns(i)
            conditionalTree = _Tree()
            conditionalTree.info = info.copy()
            for pat in range(len(patterns)):
                conditionalTree.addConditionalPatterns(patterns[pat], timeStamps[pat])
            find = conditionalTree.check()
            if find == 1:
                del patterns, timeStamps, info
                for cp in conditionalTree.generatePatterns(pattern):
                    yield cp
            else:
                if len(conditionalTree.info) != 0:
                    j = []
                    for r in timeStamps:
                        j += r
                    inf = getPeriodAndSupport(j)
                    patterns[0].reverse()
                    upp = []
                    for jm in patterns[0]:
                        upp.append(_pfList[jm])
                    allSubsets = _subLists(upp)
                    # print(upp,inf)
                    for pa in allSubsets:
                        yield pattern + pa, inf
                del patterns, timeStamps, info
                del conditionalTree
            self.removeNode(i)


def _subLists(itemSet) -> List[List[int]]:
    """
    Forms all the subsets of given itemSet

    :param itemSet: itemSet or a list of periodic-frequent items
    :return: subsets of itemSet
    """
    subs = []
    for i in range(1, len(itemSet) + 1):
        temp = [list(x) for x in _combinations(itemSet, i)]
        if len(temp) > 0:
            subs.extend(temp)

    return subs


def getPeriodAndSupport(timeStamps) -> List[int]:
    """
    Calculates the period and support of list of timeStamps

    :param timeStamps: timeStamps of a  pattern or item
    :return: support and periodicity
    """
    cur = 0
    per = 0
    sup = 0
    for j in range(len(timeStamps)):
        per = max(per, timeStamps[j].start - cur)
        per = max(per, timeStamps[j].per)
        if per > _maxPer:
            return [0, 0]
        cur = timeStamps[j].end
        sup += timeStamps[j].sup
    per = max(per, _lno - cur)
    return [sup, per]


def conditionalTransactions(patterns, timestamp) -> Tuple[List[List[int]], List[List[_Interval]], Dict[int, Tuple[int, int]]]:
    """
    To sort and update the conditional transactions by removing the items which fails frequency
    and periodicity conditions

    :param patterns: conditional patterns of a node
    :param timestamp: timeStamps of a conditional pattern
    :return: conditional transactions with their respective timeStamps
    """
    global _minSup, _maxPer
    pat = []
    timeStamps = []
    data1 = {}
    for i in range(len(patterns)):
        for j in patterns[i]:
            if j in data1:
                data1[j] = _merge(data1[j], timestamp[i])
            else:
                data1[j] = timestamp[i]

    updatedDict = {}
    for m in data1:
        updatedDict[m] = getPeriodAndSupport(data1[m])
    updatedDict = {k: v for k, v in updatedDict.items() if v[0] >= _minSup and v[1] <= _maxPer}
    count = 0
    for p in patterns:
        p1 = [v for v in p if v in updatedDict]
        trans = sorted(p1, key=lambda x: (updatedDict.get(x)[0], -x), reverse=True)
        if len(trans) > 0:
            pat.append(trans)
            timeStamps.append(timestamp[count])
        count += 1
    return pat, timeStamps, updatedDict


class PSGrowth(_ab._periodicFrequentPatterns):
    """
    :Description:   PS-Growth is one of the fundamental algorithm to discover periodic-frequent patterns in a temporal database.

    :Reference :   A. Anirudh, R. U. Kiran, P. K. Reddy and M. Kitsuregaway, "Memory efficient mining of periodic-frequent
                   patterns in transactional databases," 2016 IEEE Symposium Series on Computational Intelligence (SSCI),
                   2016, pp. 1-8, https://doi.org/10.1109/SSCI.2016.7849926

    :param  iFile: str :
                   Name of the Input file to mine complete set of periodic frequent pattern's
    :param  oFile: str :
                   Name of the output file to store complete set of periodic frequent pattern's
    :param  minSup: str:
                   Controls the minimum number of transactions in which every item must appear in a database.
    :param  maxPer: str:
                   Controls the maximum number of transactions in which any two items within a pattern can reappear.
    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.

    :Attributes:

        iFile : file
            Name of the Input file or path of the input file
        oFile : file
            Name of the output file or path of the output file
        minSup: int or float or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        maxPer: int or float or str
            The user can specify maxPer either in count or proportion of database size.
            If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxPer=10 will be treated as integer, while maxPer=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default separator is tab space or \t.
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
            it represents the total no of transaction
        tree : class
            it represents the Tree class
        itemSetCount : int
            it represents the total no of patterns
        finalPatterns : dict
            it represents to store the patterns

    :Methods:

        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        save(oFile)
            Complete set of periodic-frequent patterns will be loaded in to an output file
        getConditionalPatternsInDataFrame()
            Complete set of periodic-frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        OneLengthItems()
            Scans the dataset or dataframes and stores in list format
        buildTree()
            after updating the Databases ar added into the tree by setting root node as null


    **Methods to execute code on terminal**
    -----------------------------------------
     .. code-block:: console


       Format:

       (.venv) $ python3 PSGrowth.py <inputFile> <outputFile> <minSup> <maxPer>

       Example:

       (.venv) $ python3 PSGrowth.py sampleTDB.txt patterns.txt 0.3 0.4



               .. note:: minSup will be considered in percentage of database transactions


    **Importing this algorithm into a python program**
    ----------------------------------------------------
    .. code-block:: python

            from PAMI.periodicFrequentPattern.basic import PSGrowth as alg

            obj = alg.PSGrowth("../basic/sampleTDB.txt", "2", "6")

            obj.startMine()

            periodicFrequentPatterns = obj.getPatterns()

            print("Total number of  Patterns:", len(periodicFrequentPatterns))

            obj.save("patterns")

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

    **Credits:**
    --------------
             The complete program was written by  P.Likhitha   under the supervision of Professor Rage Uday Kiran.

    """

    _startTime = float()
    _endTime = float()
    _minSup = str()
    _maxPer = str()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _rank = {}
    _lno = 0

    def _convert(self, value) -> float:
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

    def _creatingItemSets(self) -> None:
        """
        Storing the complete transactions of the database/input file in a database variable
        :return: None
        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            ts, data = [], []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'TS' in i:
                ts = self._iFile['TS'].tolist()
            if 'Transactions' in i:
                data = self._iFile['Transactions'].tolist()
            for i in range(len(data)):
                if data[i]:
                    tr = [str(ts[i])] + [x for x in data[i].split(self._sep)]
                    self._Database.append(tr)
                else:
                    self._Database.append([str(ts[i])])

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

    def _OneLengthItems(self):
        """
        Storing the complete values of a database/input file into a database variable
        """
        data = {}
        global _minSup, _maxPer, _lno
        for tr in self._Database:
            self._lno += 1
            for i in range(1, len(tr)):
                if tr[i] not in data:
                    data[tr[i]] = [int(tr[0]), int(tr[0]), 1]
                else:
                    data[tr[i]][0] = max(data[tr[i]][0], (int(tr[0]) - data[tr[i]][1]))
                    data[tr[i]][1] = int(tr[0])
                    data[tr[i]][2] += 1
        for key in data:
            data[key][0] = max(data[key][0], self._lno - data[key][1])
        self._minSup = self._convert(self._minSup)
        self._maxPer = self._convert(self._maxPer)
        _minSup, _maxPer, _lno = self._minSup, self._maxPer, self._lno
        data = {k: [v[2], v[0]] for k, v in data.items() if v[0] <= self._maxPer and v[2] >= self._minSup}
        genList = [k for k, v in sorted(data.items(), key=lambda x: (x[1][0], x[0]), reverse=True)]
        self._rank = dict([(index, item) for (item, index) in enumerate(genList)])
        return data, genList

    def _buildTree(self, info, sampleDict) -> _Tree:
        """
        it takes the Databases and support of each item and construct the main tree with setting root node as null

        :param info: it represents the support of each item
        :type info: dictionary
        :param sampleDict: One length periodic-frequent patterns in a dictionary
        :type sampleDict: dict
        :return: Returns the root node of the tree
        """
        rootNode = _Tree()
        rootNode.info = info.copy()
        k = 0
        for line in self._Database:
            k += 1
            tr = line
            list2 = [int(tr[0])]
            for i in range(1, len(tr)):
                if tr[i] in sampleDict:
                    list2.append(self._rank[tr[i]])
            if len(list2) >= 2:
                basket = list2[1:]
                basket.sort()
                list2[1:] = basket[0:]
                rootNode.addTransaction(list2[1:], list2[0])
        return rootNode

    @deprecated("It is recommended to use mine() instead of startMine() for mining process")
    def startMine(self) -> None:
        """
        Mining process will start from this function
        :return: None
        """
        global _minSup, _maxPer, _lno, _pfList
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._minSup is None:
            raise Exception("Please enter the Minimum Support")
        self._creatingItemSets()
        OneLengthPeriodicItems, _pfList = self._OneLengthItems()
        info = {self._rank[k]: v for k, v in OneLengthPeriodicItems.items()}
        Tree = self._buildTree(info, OneLengthPeriodicItems)
        patterns = Tree.generatePatterns([])
        self._finalPatterns = {}
        for i in patterns:
            sample = str()
            for k in i[0]:
                sample = sample + k + "\t"
            self._finalPatterns[sample] = i[1]
        self._endTime = _ab._time.time()
        self._memoryUSS = float()
        self._memoryRSS = float()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Periodic-Frequent patterns were generated successfully using PS-Growth algorithm ")

    def mine(self) -> None:
        """
        Mining process will start from this function
        :return: None
        """
        global _minSup, _maxPer, _lno, _pfList
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._minSup is None:
            raise Exception("Please enter the Minimum Support")
        self._creatingItemSets()
        OneLengthPeriodicItems, _pfList = self._OneLengthItems()
        info = {self._rank[k]: v for k, v in OneLengthPeriodicItems.items()}
        Tree = self._buildTree(info, OneLengthPeriodicItems)
        patterns = Tree.generatePatterns([])
        self._finalPatterns = {}
        for i in patterns:
            sample = str()
            for k in i[0]:
                sample = sample + k + "\t"
            self._finalPatterns[sample] = i[1]
        self._endTime = _ab._time.time()
        self._memoryUSS = float()
        self._memoryRSS = float()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Periodic-Frequent patterns were generated successfully using PS-Growth algorithm ")

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

    def getPatterns(self) -> dict:
        """
        Function to send the set of periodic-frequent patterns after completion of the mining process

        :return: returning periodic-frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self)-> None:
        """
        This function is used to print the results
        :return: None
        """
        print("Total number of Periodic Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = PSGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = PSGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        _ap.mine()
        print("Total number of Periodic-Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
