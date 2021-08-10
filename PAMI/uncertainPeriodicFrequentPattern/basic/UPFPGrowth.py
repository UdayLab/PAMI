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
from PAMI.uncertainPeriodicFrequentPattern.basic.abstract import *


minSup = float()
maxPer = float()
first = int()
last = int()
lno = int()
rank = {}
periodic = {}


class Item:
    """
    A class used to represent the item with probability in transaction of dataset

        ...

        Attributes:
        __________
            item: int or word
                Represents the name of the item
            probability: float
                Represent the existential probability(likelihood presence) of an item
    """
    def __init__(self, item, probability):
        self.item = item
        self.probability = probability


class Node(object):
    """
        A class used to represent the node of frequentPatternTree

        ...

        Attributes:
        ----------
            item: int
                storing item of a node
            probability: int
                To maintain the expected support of node
            parent: node
                To maintain the parent of every node
            children: list
                To maintain the children of node
            timeStamps: list
                To maintain the timeStamps of node

        Methods:
        -------
            addChild(itemName)
                storing the children to their respective parent nodes
        """
    def __init__(self, item, children):
        self.item = item
        self.probability = 1
        self.children = children
        self.parent = None
        self.timeStamps = []

    def addChild(self, node):
        """
        To add the children details to parent node

        :param node: children node

        :return: updated parent node children
        """
        self.children[node.item] = node
        node.parent = self


def printTree(root):
    """
    To print the details of tree

    :param root: root node of the tree

    :return: details of tree
    """
    for x, y in root.children.items():
        print(x, y.item, y.probability, y.parent.item, y.timeStamps)
        printTree(y)


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
            addTransactions(transaction)
                creating transaction as a branch in frequentPatternTree
            addConditionalTransaction(prefixPaths, supportOfItems)
                construct the conditional tree for prefix paths
            conditionalPatterns(Node)
                generates the conditional patterns from tree for specific node
            conditionalTransactions(prefixPaths,Support)
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

    def addTransactions(self, transaction, tid):
        """adding transaction into tree

            :param transaction: it represents the one transactions in database

            :type transaction: list

            :param tid: the timestamp of transaction

            :type tid: list
        """
        currentNode = self.root
        for i in range(len(transaction)):
            if transaction[i].item not in currentNode.children:
                newNode = Node(transaction[i].item, {})
                l1 = i - 1
                temp = []
                while l1 >= 0:
                    temp.append(transaction[l1].probability)
                    l1 -= 1
                if len(temp) == 0:
                    newNode.probability = transaction[i].probability
                else:
                    newNode.probability = max(temp) * transaction[i].probability
                currentNode.addChild(newNode)
                if transaction[i].item in self.summaries:
                    self.summaries[transaction[i].item].append(newNode)
                else:
                    self.summaries[transaction[i].item] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i].item]
                l1 = i - 1
                temp = []
                while l1 >= 0:
                    temp.append(transaction[l1].probability)
                    l1 -= 1
                if len(temp) == 0:
                    currentNode.probability += transaction[i].probability
                else:
                    currentNode.probability += max(temp) * transaction[i].probability
        currentNode.timeStamps = currentNode.timeStamps + tid

    def addConditionalTransaction(self, transaction, ts, sup):
        """constructing conditional tree from prefixPaths

                :param transaction : it represents the one transactions in database

                :type transaction : list

                :param ts: timeStamp of a transaction

                :type ts: list

                :param sup : support of prefixPath taken at last child of the path

                :type sup : int
        """
        currentNode = self.root
        for i in range(len(transaction)):
            if transaction[i] not in currentNode.children:
                newNode = Node(transaction[i], {})
                newNode.probability = sup
                currentNode.addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]
                currentNode.probability += sup
        currentNode.timeStamps = currentNode.timeStamps + ts

    def getConditionalPatterns(self, alpha):
        """generates all the conditional patterns of respective node

            :param alpha : it represents the Node in tree

            :type alpha : Node
        """

        finalPatterns = []
        finalTimeStamps = []
        sup = []
        for i in self.summaries[alpha]:
            set1 = i.timeStamps
            s = i.probability
            set2 = []
            while i.parent.item is not None:
                set2.append(i.parent.item)
                i = i.parent
            if len(set2) > 0:
                set2.reverse()
                finalPatterns.append(set2)
                finalTimeStamps.append(set1)
                sup.append(s)
        finalPatterns, finalTimeStamps, support, info = self.conditionalTransactions(finalPatterns, finalTimeStamps, sup)
        return finalPatterns, finalTimeStamps, support, info

    def removeNode(self, nodeValue):
        """removing the node from tree

            :param nodeValue : it represents the node in tree

            :type nodeValue : node
        """
        for i in self.summaries[nodeValue]:
            i.parent.timeStamps = i.parent.timeStamps + i.timeStamps
            del i.parent.children[nodeValue]

    def getPeriodAndSupport(self, s, timeStamps):
        global lno, maxPer
        timeStamps.sort()
        cur = 0
        per = 0
        sup = s
        for j in range(len(timeStamps)):
            per = max(per, timeStamps[j] - cur)
            if per > maxPer:
                return [0, 0]
            cur = timeStamps[j]
        per = max(per, lno - cur)
        return [sup, per]

    def conditionalTransactions(self, condPatterns, condTimeStamps, support):
        """ It generates the conditional patterns with frequent items

                :param condPatterns : conditional patterns generated from getConditionalPatterns method for respective node

                :type condPatterns : list

                :param condTimeStamps: timeStamps of conditional transactions

                :type condTimeStamps: list

                :param support : the support of conditional pattern in tree

                :type support : list
        """
        global minSup, maxPer
        pat = []
        timeStamps = []
        sup = []
        data1 = {}
        count = {}
        for i in range(len(condPatterns)):
            for j in condPatterns[i]:
                if j in data1:
                    data1[j] = data1[j] + condTimeStamps[i]
                    count[j] += support[i]
                else:
                    data1[j] = condTimeStamps[i]
                    count[j] = support[i]
        updatedDict = {}
        for m in data1:
            updatedDict[m] = self.getPeriodAndSupport(count[m], data1[m])
        updatedDict = {k: v for k, v in updatedDict.items() if v[0] >= minSup and v[1] <= maxPer}
        count = 0
        for p in condPatterns:
            p1 = [v for v in p if v in updatedDict]
            trans = sorted(p1, key=lambda x: (updatedDict.get(x)[0]), reverse=True)
            if len(trans) > 0:
                pat.append(trans)
                timeStamps.append(condTimeStamps[count])
                sup.append(support[count])
            count += 1
        return pat, timeStamps, sup, updatedDict

    def generatePatterns(self, prefix):
        """generates the patterns

            :param prefix : forms the combination of items

            :type prefix : list
        """

        global periodic, minSup
        for i in sorted(self.summaries, key=lambda x: (self.info.get(x)[0])):
            pattern = prefix[:]
            pattern.append(i)
            s = 0
            for x in self.summaries[i]:
                s += x.probability
            if s >= minSup:
                periodic[tuple(pattern)] = self.info[i]
                patterns, timeStamps, support, info = self.getConditionalPatterns(i)
                conditionalTree = Tree()
                conditionalTree.info = info.copy()
                for pat in range(len(patterns)):
                    conditionalTree.addConditionalTransaction(patterns[pat], timeStamps[pat], support[pat])
                if len(patterns) > 0:
                    conditionalTree.generatePatterns(pattern)
            self.removeNode(i)


class UPFPGrowth(periodicFrequentPatterns):
    """

        UPFPGrowth is  to discover periodic-frequent patterns in a temporal database.

        Reference:
        --------

        Attributes:
        ----------
            iFile : file
                Name of the Input file or path of the input file
            oFile : file
                Name of the output file or path of output file
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
            sep: str
                This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
                However, the users can override their default separator.
            memoryUSS: float
                To store the total amount of USS memory consumed by the program
            memoryRSS: float
                To store the total amount of RSS memory consumed by the program
            startTime: float
                To record the start time of the mining process
            endTime: float
                To record the completion time of the mining process
            Database : list
                To store the transactions of a database in list
            mapSupport : Dictionary
                To maintain the information of item and their frequency
            lno : int
                To represent the total no of transaction
            tree : class
                To represents the Tree class
            finalPatterns : dict
                To store the complete patterns

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
            creatingItemSets()
                Scans the dataset and stores in a list format
            PeriodicFrequentOneItem()
                Extracts the one-periodic-frequent patterns from database
            updateTransaction()
                Update the database by removing aperiodic items and sort the Database by item decreased support
            buildTree()
                After updating the Database, remaining items will be added into the tree by setting root node as null
            convert()
                to convert the user specified value
            removeFalsePositives()
                to remove the false positives in generated patterns

        Executing the code on terminal:
        -------
            Format:
            ------
                python3 UPFPGrowth.py <inputFile> <outputFile> <minSup> <maxPer>
            Examples:
            --------
                python3 UPFPGrowth.py sampleTDB.txt patterns.txt 0.3 4     (minSup and maxPer will be considered in support count or frequency)

        Sample run of importing the code:
        -------------------
        
            from PAMI.uncertainPeriodicFrequentPattern import UPFPGrowth as alg
        
            obj = alg.UPFPGrowth(iFile, minSup, maxPer)
        
            obj.startMine()
        
            periodicFrequentPatterns = obj.getPatterns()
        
            print("Total number of Periodic Frequent Patterns:", len(periodicFrequentPatterns))
        
            obj.storePatternsInFile(oFile)
        
            Df = obj.getPatternsInDataFrame()
        
            memUSS = obj.getMemoryUSS()
        
            print("Total Memory in USS:", memUSS)
        
            memRSS = obj.getMemoryRSS()
        
            print("Total Memory in RSS", memRSS)
        
            run = obj.getRuntime()
        
            print("Total ExecutionTime in seconds:", run)
        
    Credits:
    -------
        The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.
        
    """
    rank = {}
    startTime = float()
    endTime = float()
    minSup = float()
    maxPer = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    sep = " "
    memoryUSS = float()
    memoryRSS = float()
    Database = []
    lno = 0

    def creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable

        """
        try:
            with open(self.iFile, 'r') as f:
                for line in f:
                    temp = [i.rstrip() for i in line.split(self.sep)]
                    temp = [x for x in temp if x]
                    tr = [int(temp[0])]
                    for i in temp[1:]:
                        i1 = i.index('(')
                        i2 = i.index(')')
                        item = i[0:i1]
                        probability = float(i[i1 + 1:i2])
                        product = Item(item, probability)
                        tr.append(product)
                    self.lno += 1
                    self.Database.append(tr)
        except IOError:
            print("File Not Found")

    def periodicFrequentOneItem(self):
        """takes the transactions and calculates the support of each item in the dataset and assign the
                    ranks to the items by decreasing support and returns the frequent items list

        """
        mapSupport = {}
        for i in self.Database:
            n = i[0]
            for j in i[1:]:
                if j.item not in mapSupport:
                    mapSupport[j.item] = [j.probability, abs(0 - n), n]
                else:
                    mapSupport[j.item][0] += j.probability
                    mapSupport[j.item][1] = max(mapSupport[j.item][1], abs(n - mapSupport[j.item][2]))
                    mapSupport[j.item][2] = n
        for key in mapSupport:
            mapSupport[key][1] = max(mapSupport[key][1], len(self.Database) - mapSupport[key][2])
        mapSupport = {k: [v[0], v[1]] for k, v in mapSupport.items() if v[1] <= self.maxPer and v[0] >= self.minSup}
        plist = [k for k, v in sorted(mapSupport.items(), key=lambda x: (x[1][0], x[0]), reverse=True)]
        self.rank = dict([(index, item) for (item, index) in enumerate(plist)])
        return mapSupport, plist

    def check(self, i, x):
        """To check the presence of item or pattern in transaction

            :param x: it represents the pattern

            :type x : list

            :param i : represents the uncertain transactions

            :type i : list
        """

        for m in x:
            k = 0
            for n in i:
                if m == n.item:
                    k += 1
            if k == 0:
                return 0
        return 1

    def getPeriodAndSupport(self, s, timeStamps):
        """
        To calculate periodicity of timeStamps

            :param s: support of a pattern

            :param timeStamps: timeStamps of a pattern

            :return: periodicity and Support
        """
        global lno, maxPer
        timeStamps.sort()
        cur = 0
        per = 0
        sup = s
        for j in range(len(timeStamps)):
            per = max(per, timeStamps[j] - cur)
            if per > maxPer:
                return [0, 0]
            cur = timeStamps[j]
        per = max(per, lno - cur)
        return [sup, per]

    def buildTree(self, data, info):
        """it takes the transactions and support of each item and construct the main tree with setting root
                    node as null

            :param data: it represents the one transactions in database

            :type data: list

            :param info: it represents the support of each item

            :type info : dictionary
        """
        rootNode = Tree()
        rootNode.info = info.copy()
        for i in range(len(data)):
            set1 = [data[i][0]]
            rootNode.addTransactions(data[i][1:], set1)
        return rootNode

    def updateTransactions(self, dict1):
        """remove the items which are not frequent from transactions and updates the transactions with rank of items

            :param dict1 : frequent items with support

            :type dict1 : dictionary
        """

        list1 = []
        for tr in self.Database:
            list2 = [int(tr[0])]
            for i in range(1, len(tr)):
                if tr[i].item in dict1:
                    list2.append(tr[i])
            if len(list2) >= 2:
                basket = list2[1:]
                basket.sort(key=lambda val: self.rank[val.item])
                list2[1:] = basket[0:]
                list1.append(list2)
        return list1

    def convert(self, value):
        """
            To convert the given user specified value

            :param value: user specified value

            :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = float(value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)

        return value
    
    def removeFalsePositives(self):
        """

        Returns
        -------
            removes the false positive patterns in generated patterns
        """
        periods = {}
        for i in self.Database:
            for x, y in periodic.items():
                if len(x) == 1:
                    periods[x] = y
                else:
                    s = 1
                    check = self.check(i[1:], x)
                    if check == 1:
                        for j in i[1:]:
                            if j.item in x:
                                s *= j.probability
                        if x in periods:
                            periods[x][0] += s
                        else:
                            periods[x] = [s, y[1]]
        for x, y in periods.items():
            if y[0] >= minSup:
                sample = str()
                for i in x:
                    sample = sample + i + " " 
                self.finalPatterns[sample] = y

    def startMine(self):
        """Main method where the patterns are mined by constructing tree and remove the remove the false patterns
                    by counting the original support of a patterns


        """
        global lno, maxPer, minSup, first, last, periodic
        minSup = self.minSup
        maxPer = self.maxPer
        self.startTime = time.time()
        lno = 1
        self.creatingItemSets()
        self.minSup = self.convert(self.minSup)
        self.maxPer = self.convert(self.maxPer)
        minSup, maxPer, lno = self.minSup, self.maxPer, self.lno
        mapSupport, plist = self.periodicFrequentOneItem()
        updatedTrans = self.updateTransactions(mapSupport)
        info = {k: v for k, v in mapSupport.items()}
        Tree1 = self.buildTree(updatedTrans, info)
        Tree1.generatePatterns([])
        self.removeFalsePositives()
        print("Periodic frequent patterns were generated successfully using UPFP algorithm")
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss

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

        dataframe = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b[0], b[1]])
            dataframe = pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
        return dataframe

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
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:
            ap = UPFPGrowth(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 5:
            ap = UPFPGrowth(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Patterns:", len(Patterns))
        ap.storePatternsInFile(sys.argv[2])
        #print(ap.getPatternsInDataFrame())
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
