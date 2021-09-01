#  Copyright (C)  2021 Rage Uday Kiran
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from PAMI.frequentPattern.maximal.abstract import *


pfList = []
patterns = {}
minSup = str()


class Node(object):
    """ A class used to represent the node of frequentPatternTree


        Attributes:
        ----------
            item : int
                storing item of a node
            counter : list
                To maintain the support of the node
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
        """ Initializing the Node class

        :param item: Storing the item of a node

        :type item: int or None

        :param children: To maintain the children of a node

        :type children: dict
        """
        self.item = item
        self.children = children
        self.counter = int()
        self.parent = None

    def addChild(self, node):
        """Adding a child to the created node

        :param node: node object

        :type node: object
        """
        self.children[node.item] = node
        node.parent = self


class Tree(object):
    """
        A class used to represent the frequentPatternGrowth tree structure


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
            addTransaction(transaction)
                creating transaction as a branch in frequentPatternTree
            addConditionalTransaction(prefixPaths, supportOfItems)
                construct the conditional tree for prefix paths
            condPatterns(Node)
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
        self.root = Node(None, {})
        self.summaries = {}
        self.info = {}

    def addTransaction(self, transaction):
        """
        adding transactions into tree

        :param transaction: represents the transaction in a database

        :return: tree
        """
        currentNode = self.root
        for i in range(len(transaction)):
            if transaction[i] not in currentNode.children:
                newNode = Node(transaction[i], {})
                newNode.counter = 1
                currentNode.addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]
                currentNode.counter += 1

    def addConditionalTransaction(self, transaction, count):
        """
            Loading the database into a tree

        :param transaction: conditional transaction of a node

        :param count: the support of conditional transaction

        :return: conditional tree
        """
        currentNode = self.root
        for i in range(len(transaction)):
            if transaction[i] not in currentNode.children:
                newNode = Node(transaction[i], {})
                newNode.counter = count
                currentNode.addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]
                currentNode.counter += count

    def getConditionalPatterns(self, alpha):
        """
        generates all the conditional patterns of respective node

        :param alpha: it represents the Node in tree

        :return: conditional patterns of a node
        """
        finalPatterns = []
        finalSets = []
        for i in self.summaries[alpha]:
            set1 = i.counter
            set2 = []
            while i.parent.item is not None:
                set2.append(i.parent.item)
                i = i.parent
            if len(set2) > 0:
                set2.reverse()
                finalPatterns.append(set2)
                finalSets.append(set1)
        finalPatterns, finalSets, info = self.conditionalTransactions(finalPatterns, finalSets)
        return finalPatterns, finalSets, info

    def conditionalTransactions(self, condPatterns, condFreq):
        """
        sorting and removing the items from conditional transactions which don't satisfy minSup

        :param condPatterns: conditional patterns if a node

        :param condFreq: frequency at leaf node of conditional transaction

        :return: conditional patterns and their frequency respectively
        """
        global minSup
        pat = []
        tids = []
        data1 = {}
        for i in range(len(condPatterns)):
            for j in condPatterns[i]:
                if j not in data1:
                    data1[j] = condFreq[i]
                else:
                    data1[j] += condFreq[i]
        updatedDict = {}
        updatedDict = {k: v for k, v in data1.items() if v >= minSup}
        count = 0
        for p in condPatterns:
            p1 = [v for v in p if v in updatedDict]
            trans = sorted(p1, key=lambda x: (updatedDict.get(x), -x), reverse=True)
            if len(trans) > 0:
                pat.append(trans)
                tids.append(condFreq[count])
            count += 1
        return pat, tids, updatedDict

    def removeNode(self, nodeValue):
        """
        to remove the node from the original tree

        :param nodeValue: leaf node of tree

        :return: tree after deleting node
        """
        for i in self.summaries[nodeValue]:
            del i.parent.children[nodeValue]
            i = None

    def generatePatterns(self, prefix):
        """
        generates the patterns

        :param prefix: forms the combination of items

        :return: the maximal frequent patterns
        """
        global maximalTree, pfList, patterns
        for i in sorted(self.summaries, key=lambda x: (self.info.get(x), -x)):
            pattern = prefix[:]
            pattern.append(i)
            condPatterns, tids, info = self.getConditionalPatterns(i)
            conditional_tree = Tree()
            conditional_tree.info = info.copy()
            head = pattern[:]
            tail = []
            for la in info:
                tail.append(la)
            sub = head + tail
            if maximalTree.checkerSub(sub) == 1:
                for pat in range(len(condPatterns)):
                    conditional_tree.addConditionalTransaction(condPatterns[pat], tids[pat])
                if len(condPatterns) >= 1:
                    conditional_tree.generatePatterns(pattern)
                else:
                    maximalTree.addTransaction(pattern)
                    s = convertItems(pattern)
                    patterns[tuple(s)] = self.info[i]
            self.removeNode(i)


def convertItems(itemSet):
    """
    To convert the item ranks into their original item names

    :param itemSet: itemSet or a pattern

    :return: original pattern
    """
    t1 = []
    for i in itemSet:
        t1.append(pfList[i])
    return t1


class MNode(object):
    """
        A class used to represent the node in maximal tree

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

    def addChild(self, node):
        """
        To add the children details to a parent node

        :param node: children node

        :return: adding children details to parent node
        """
        self.children[node.item] = node
        node.parent = self


class MPTree(object):
    """
        A class used to represent the frequentPatternGrowth tree structure

        Attributes:
        ----------
            root : Node
                Represents the root node of the tree
            summaries : dictionary
                storing the nodes with same item name


            Methods
            -------
            addTransaction(transaction)
                creating transaction as a branch in frequentPatternTree
            addConditionalTransaction(prefixPaths, supportOfItems)
                construct the conditional tree for prefix paths
            checkerSub(items):
                Given a set of items to the subset of them is present or not
    """

    def __init__(self):
        self.root = Node(None, {})
        self.summaries = {}

    def addTransaction(self, transaction):
        """
        To construct the maximal frequent pattern into maximal tree

        :param transaction: the maximal frequent patterns extracted till now

        :return: the maximal tree
        """
        currentNode = self.root
        transaction.sort()
        for i in range(len(transaction)):
            if transaction[i] not in currentNode.children:
                newNode = MNode(transaction[i], {})
                currentNode.addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].insert(0, newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]

    def checkerSub(self, items):
        """
        To check the subset of pattern present in tree

        :param items: the sub frequent pattern

        :return: checks if subset present in the tree
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


# Initialising the  variable for maximal tree
maximalTree = MPTree()


class Maxfpgrowth(frequentPatterns):
    """
    MaxFP-Growth is one of the fundamental algorithm to discover maximal frequent patterns in a transactional database.

    Reference:
    ---------
        Grahne, G. and Zhu, J., "High Performance Mining of Maximal Frequent itemSets",
        http://users.encs.concordia.ca/~grahne/papers/hpdm03.pdf

    Attributes:
    ----------
        iFile : file
            Name of the Input file to mine complete set of frequent patterns
        oFile : file
            Name of the output file to store complete set of frequent patterns
        minSup: float or int or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
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
            it represents the total no of transaction
        tree : class
            it represents the Tree class
        finalPatterns : dict
            it represents to store the patterns

    Methods:
    -------
        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        storePatternsInFile(oFile)
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsInDataFrame()
            Complete set of frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        creatingItemSets()
            Scans the dataset or dataframes and stores in list format
        frequentOneItem()
            Extracts the one-frequent patterns from Databases
        updateTransactions()
            update the Databases by removing aperiodic items and sort the Database by item decreased support
        buildTree()
            after updating the Databases ar added into the tree by setting root node as null
        startMine()
            the main method to run the program

    Executing the code on terminal:
    -------
        Format:
        ------
            python3 maxfpgrowth.py <inputFile> <outputFile> <minSup>

        Examples:
        -------
            python3 maxfpgrowth.py sampleDB.txt patterns.txt 0.3   (minSup will be considered in percentage of database transactions)

            python3 maxfpgrowth.py sampleDB.txt patterns.txt 3     (minSup will be considered in support count or frequency)

    Sample run of the imported code:
    --------------
        from PAMI.frequentPattern.maximal import maxfpgrowth as alg

        obj = alg.Maxfpgrowth("../basic/sampleTDB.txt", "2")

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.storePatternsInFile("patterns")

        Df = obj.getPatternsInDataFrame()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    Credits:
    -------
        The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.\n

    """
    startTime = float()
    endTime = float()
    minSup = str()
    maxPer = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    sep = " "
    memoryUSS = float()
    memoryRSS = float()
    Database = []
    rank = {}
    rankdup = {}
    lno = 0

    def creatingItemSets(self):
        """
        Storing the complete Databases of the database/input file in a database variable
        """
        try:
            with open(self.iFile, 'r', encoding='utf-8') as f:
                for line in f:
                    li = [i.rstrip() for i in line.split(self.sep)]
                    self.Database.append(li)
                    self.lno += 1
        except IOError:
            print("File Not Found")

    def frequentOneItem(self):
        """ To extract the one-length frequent itemSets

        :return: 1-length frequent items
        """
        global pfList
        mapSupport = {}
        k = 0
        for tr in self.Database:
            k += 1
            for i in range(0, len(tr)):
                if tr[i] not in mapSupport:
                    mapSupport[tr[i]] = 1
                else:
                    mapSupport[tr[i]] += 1
        mapSupport = {k: v for k, v in mapSupport.items() if v >= self.minSup}
        genList = [k for k, v in sorted(mapSupport.items(), key=lambda x: x[1], reverse=True)]
        self.rank = dict([(index, item) for (item, index) in enumerate(genList)])
        return mapSupport, genList

    def updateTransactions(self, oneLength):
        """ To sort the transactions in their support descending order and allocating ranks respectively

        :param oneLength: 1-length frequent items in dictionary

        :return: returning the sorted list

        :Example: oneLength = {'a':7, 'b': 5, 'c':'4', 'd':3}
                    rank = {'a':0, 'b':1, 'c':2, 'd':3}
        """
        list1 = []
        for tr in self.Database:
            list2 = []
            for i in range(0, len(tr)):
                if tr[i] in oneLength:
                    list2.append(self.rank[tr[i]])
            if len(list2) >= 2:
                list2.sort()
                list1.append(list2)
        return list1

    @staticmethod
    def buildTree(data, info):
        """
        creating the root node as null in fp-tree and and adding all transactions into tree.
        :param data: updated transactions
        :param info: rank of items in transactions
        :return: fp-tree
        """
        rootNode = Tree()
        rootNode.info = info.copy()
        for i in range(len(data)):
            rootNode.addTransaction(data[i])
        return rootNode
    

    def convert(self, value):
        """
        to convert the type of user specified minSup value
        :param value: user specified minSup value
        :return: converted type
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self.lno * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self.lno * value)
            else:
                value = int(value)
        return value

    def startMine(self):
        """
                Mining process will start from this function
        """

        global minSup, patterns, pfList
        self.startTime = time.time()
        if self.iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self.minSup is None:
            raise Exception("Please enter the Minimum Support")
        self.creatingItemSets()
        self.minSup = self.convert(self.minSup)
        minSup = self.minSup
        generatedItems, pfList = self.frequentOneItem()
        updatedTransactions = self.updateTransactions(generatedItems)
        for x, y in self.rank.items():
            self.rankdup[y] = x
        info = {self.rank[k]: v for k, v in generatedItems.items()}
        Tree = self.buildTree(updatedTransactions, info)
        Tree.generatePatterns([])
        for x, y in patterns.items():
            pattern = str()
            for i in x:
                pattern = pattern + i + " "
            self.finalPatterns[pattern] = y
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Maximal Frequent patterns were generated successfully using MaxFp-Growth algorithm ")

    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process

        :rtype: float
        """

        return self.memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process

        :rtype: float
        """

        return self.memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process

        :return: returning total amount of runtime taken by the mining process

        :rtype: float
        """

        return self.endTime - self.startTime

    def getPatternsInDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def storePatternsInFile(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            s1 = x + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns

        :rtype: dict
        """
        return self.finalPatterns


if __name__ == "__main__":
    ap = str()
    if len(sys.argv) == 4 or len(sys.argv) == 5:
        if len(sys.argv) == 5:
            ap = Maxfpgrowth(sys.argv[1], sys.argv[3], sys.argv[4])
        if len(sys.argv) == 4:
            ap = Maxfpgrowth(sys.argv[1], sys.argv[3])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Maximal Frequent Patterns:", len(Patterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
