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
from PAMI.periodicFrequentPattern.maximal.abstract import *


minSup = float()
maxPer = float()
lno = int()
patterns = {}
pfList = []


class Node(object):
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

    def addChild(self, node):
        """
        To add the children details to the parent node children list

        :param node: children node

        :return: adding to parent node children
        """
        self.children[node.item] = node
        node.parent = self


class Tree(object):
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
        self.root = Node(None, {})
        self.summaries = {}
        self.info = {}

    def addTransaction(self, transaction, tid):
        """
        adding transaction into database

        :param transaction: transactions in a database

        :param tid: timestamp of the transaction in database

        :return: pftree
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
        currentNode.timeStamps = currentNode.timeStamps + tid

    def getConditionalPatterns(self, alpha):
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
        finalPatterns, finalSets, info = conditionalTransactions(finalPatterns, finalSets)
        return finalPatterns, finalSets, info

    def removeNode(self, nodeValue):
        """
        removes the leaf node by pushing its timestamps to parent node

        :param nodeValue: node of a tree

        """
        for i in self.summaries[nodeValue]:
            i.parent.timeStamps = i.parent.timeStamps + i.timeStamps
            del i.parent.children[nodeValue]
            i = None

    def getTimeStamps(self, alpha):
        """
        to get all the timestamps related to a node in tree

        :param alpha: node of a tree

        :return: timestamps of a node
        """
        temp = []
        for i in self.summaries[alpha]:
            temp += i.timeStamps
        return temp

    def generatePatterns(self, prefix):
        """
            To generate the maximal periodic frequent patterns

            :param prefix: an empty list of itemSet to form the combinations

            :return: maximal periodic frequent patterns
        """
        global maximalTree, patterns
        for i in sorted(self.summaries, key=lambda x: (self.info.get(x), -x)):
            pattern = prefix[:]
            pattern.append(i)
            condPattern, timeStamps, info = self.getConditionalPatterns(i)
            conditionalTree = Tree()
            conditionalTree.info = info.copy()
            head = pattern[:]
            tail = []
            for k in info:
                tail.append(k)
            sub = head + tail
            if maximalTree.checkerSub(sub) == 1:
                for pat in range(len(condPattern)):
                    conditionalTree.addTransaction(condPattern[pat], timeStamps[pat])
                if len(condPattern) >= 1:
                    conditionalTree.generatePatterns(pattern)
                else:
                    maximalTree.addTransaction(pattern)
                    s = convert(pattern)
                    patterns[tuple(s)] = self.info[i]
            self.removeNode(i)


def convert(itemSet):
    """
    to convert the maximal pattern items with their original item names

    :param itemSet: maximal periodic frequent pattern

    :return: pattern with original item names
    """
    t1 = []
    for i in itemSet:
        t1.append(pfList[i])
    return t1


class MNode(object):
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

    def addChild(self, node):
        """
        To add the children details to parent node children variable

        :param node: children node

        :return: adding children node to parent node
        """
        self.children[node.item] = node
        node.parent = self


class MPTree(object):
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
            addTransaction(itemSet)
                the genarated periodic-frequent pattern is added into maximal-tree
            checkerSub(itemSet)
                to check of subset of itemSet is present in tree
    """
    def __init__(self):
        self.root = Node(None, {})
        self.summaries = {}

    def addTransaction(self, transaction):
        """
        to add the transaction in maximal tree
        :param transaction: resultant periodic frequent pattern
        :return: maximal tree
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


maximalTree = MPTree()


def getPeriodAndSupport(timeStamps):
    """
    To calculate the periodicity and support of a pattern with their respective timeStamps
    :param timeStamps: timeStamps
    :return: Support and periodicity
    """
    timeStamps.sort()
    cur = 0
    per = 0
    sup = 0
    for j in range(len(timeStamps)):
        per = max(per, timeStamps[j] - cur)
        if per > maxPer:
            return [0, 0]
        cur = timeStamps[j]
        sup += 1
    per = max(per, abs(lno - cur))
    return [sup, per]


def conditionalTransactions(condPatterns, condTimeStamps):
    """
    To calculate the timestamps of conditional items in conditional patterns
    :param condPatterns: conditional patterns of node
    :param condTimeStamps: timeStamps of a conditional patterns
    :return: removing items with low minSup or periodicity and sort the conditional transactions
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
        updatedDict[m] = getPeriodAndSupport(data1[m])
    updatedDict = {k: v for k, v in updatedDict.items() if v[0] >= minSup and v[1] <= maxPer}
    count = 0
    for p in condPatterns:
        p1 = [v for v in p if v in updatedDict]
        trans = sorted(p1, key=lambda x: (updatedDict.get(x)[0], -x), reverse=True)
        if len(trans) > 0:
            pat.append(trans)
            timeStamps.append(condTimeStamps[count])
        count += 1
    return pat, timeStamps, updatedDict


class Maxpfgrowth(periodicFrequentPatterns):
    """ MaxPF-Growth is one of the fundamental algorithm to discover maximal periodic-frequent
        patterns in a temporal database.

        Reference:
        --------
            R. Uday Kiran, Yutaka Watanobe, Bhaskar Chaudhury, Koji Zettsu, Masashi Toyoda, Masaru Kitsuregawa,
            "Discovering Maximal Periodic-Frequent Patterns in Very Large Temporal Databases",
            IEEE 2020, https://ieeexplore.ieee.org/document/9260063

        Attributes:
        ----------
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
            itemSetCount : int
                it represents the total no of patterns
            finalPatterns : dict
                it represents to store the patterns

        Methods:
        -------
            startMine()
                Mining process will start from here
            getPatterns()
                Complete set of patterns will be retrieved with this function
            storePatternsInFile(oFile)
                Complete set of periodic-frequent patterns will be loaded in to a output file
            getPatternsInDataFrame()
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
        -------
            Format:
            ------
            python3 maxpfrowth.py <inputFile> <outputFile> <minSup> <maxPer>

            Examples:
            --------
            python3 maxpfrowth.py sampleTDB.txt patterns.txt 0.3 0.4  (minSup will be considered in percentage of database
            transactions)
            python3 maxpfrowth.py sampleTDB.txt patterns.txt 3 4  (minSup will be considered in support count or frequency)
            
            
        Sample run of the imported code:
        --------------
            from PAMI.periodicFrequentPattern.maximal import maxpfgrowth as alg

            obj = alg.Maxpfgrowth("../basic/sampleTDB.txt", "2", "6")

            obj.startMine()

            Patterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(Patterns))

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
    rankedUp = {}
    lno = 0

    def creatingitemSets(self):
        """ Storing the complete Databases of the database/input file in a database variable
            :rtype: storing transactions into Database variable
        """

        try:
            with open(self.iFile, 'r', encoding='utf-8') as f:
                for line in f:
                    li = line.split(self.sep)
                    li1 = [i.strip() for i in li]
                    self.Database.append(li1)
                    self.lno += 1
        except IOError:
            print("File Not Found")

    def periodicFrequentOneItem(self):
        """
            calculates the support of each item in the dataset and assign the ranks to the items
            by decreasing support and returns the frequent items list
            :rtype: return the one-length periodic frequent patterns


            """
        global pfList
        data = {}
        for tr in self.Database:
            for i in range(1, len(tr)):
                if tr[i] not in data:
                    data[tr[i]] = [int(tr[0]), int(tr[0]), 1]
                else:
                    data[tr[i]][0] = max(data[tr[i]][0], (int(tr[0]) - data[tr[i]][1]))
                    data[tr[i]][1] = int(tr[0])
                    data[tr[i]][2] += 1
        for key in data:
            data[key][0] = max(data[key][0], abs(len(self.Database) - data[key][1]))
        data = {k: [v[2], v[0]] for k, v in data.items() if v[0] <= self.maxPer and v[2] >= self.minSup}
        pfList = [k for k, v in sorted(data.items(), key=lambda x: (x[1][0], x[0]), reverse=True)]
        self.rank = dict([(index, item) for (item, index) in enumerate(pfList)])
        return data

    def updateDatabases(self, dict1):
        """ Remove the items which are not frequent from Databases and updates the Databases with rank of items

            :param dict1: frequent items with support
            :type dict1: dictionary
            :rtype: sorted and updated transactions
            """
        list1 = []
        for tr in self.Database:
            list2 = [int(tr[0])]
            for i in range(1, len(tr)):
                if tr[i] in dict1:
                    list2.append(self.rank[tr[i]])
            if len(list2) >= 2:
                basket = list2[1:]
                basket.sort()
                list2[1:] = basket[0:]
                list1.append(list2)
        return list1

    @staticmethod
    def buildTree(data, info):
        """ it takes the Databases and support of each item and construct the main tree with setting root node as null

            :param data: it represents the one Databases in database
            :type data: list
            :param info: it represents the support of each item
            :type info: dictionary
            :rtype: returns root node of tree
        """

        rootNode = Tree()
        rootNode.info = info.copy()
        for i in range(len(data)):
            set1 = [data[i][0]]
            rootNode.addTransaction(data[i][1:], set1)
        return rootNode

    def savePeriodic(self, itemSet):
        """
        To convert the ranks of items in to their original item names
        :param itemSet: frequent pattern
        :return: frequent pattern with original item names
        """
        t1 = []
        for i in itemSet:
            t1.append(self.rankedUp[i])
        return t1

    def convert(self, value):
        """
        To convert the given user specified value

        :param value: user specified value
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self.Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self.Database) * value)
            else:
                value = int(value)
        return value

    
    def startMine(self):
        """ Mining process will start from this function
        """

        global minSup, maxPer, lno, patterns
        self.startTime = time.time()
        if self.iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self.minSup is None:
            raise Exception("Please enter the Minimum Support")
        self.creatingitemSets()
        self.minSup = self.convert(self.minSup)
        self.maxPer = self.convert(self.maxPer)
        minSup, maxPer, lno = self.minSup, self.maxPer, len(self.Database)
        if self.minSup > len(self.Database):
            raise Exception("Please enter the minSup in range between 0 to 1")
        generatedItems = self.periodicFrequentOneItem()
        updatedDatabases = self.updateDatabases(generatedItems)
        for x, y in self.rank.items():
            self.rankedUp[y] = x
        info = {self.rank[k]: v for k, v in generatedItems.items()}
        Tree = self.buildTree(updatedDatabases, info)
        Tree.generatePatterns([])
        for x,y in patterns.items():
            sample = str()
            for i in x:
                sample = sample + i + " "
            self.finalPatterns[sample] = y
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Maximal Periodic Frequent patterns were generated successfully using MAX-PFPGrowth algorithm ")

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
        """Storing final periodic-frequent patterns in a dataframe

        :return: returning periodic-frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b[0], b[1]])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
        return dataFrame

    def storePatternsInFile(self, outFile):
        """Complete set of periodic-frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            s1 = x + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of periodic-frequent patterns after completion of the mining process

        :return: returning periodic-frequent patterns
        :rtype: dict
        """
        return self.finalPatterns


if __name__ == "__main__":
    ap = str()
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:
            ap = Maxpfgrowth(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 5:
            ap = Maxpfgrowth(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of  Patterns:", len(Patterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
        
