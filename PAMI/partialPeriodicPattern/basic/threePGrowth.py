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

from PAMI.partialPeriodicPattern.basic.abstract import *
import sys

periodicSupport = float()
period = float()
lno = int()


class Node(object):
    """
        A class used to represent the node of frequentPatternTree
        ...
        Attributes
        ----------
        item : int
            storing item of a node
        timeStamps : list
            To maintain the timestamps of transaction at the end of the branch
        parent : node
            To maintain the parent of every node
        children : list
            To maintain the children of node

        Methods
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
        self.children[node.item] = node
        node.parent = self


class Tree(object):
    """
        A class used to represent the frequentPatternGrowth tree structure

        ...

        Attributes
        ----------
        root : Node
            Represents the root node of the tree
        summaries : dictionary
            storing the nodes with same item name
        info : dictionary
            stores the support of items


        Methods
        -------
        addTransaction(transaction)
            creating transaction as a branch in frequentPatternTree
        getConditionalPatterns(Node)
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

    def addTransaction(self, transaction, tid):
        """
                adding transaction into tree

                :param transaction : it represents the one transactions in database
                :type transaction : list
                :param tid : represents the timestamp of transaction
                :type tid : list
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
            generates all the conditional patterns of respective node

            :param alpha : it represents the Node in tree
            :type alpha : Node
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
        finalPatterns, finalSets, info = self.conditionalTransactions(finalPatterns, finalSets)
        return finalPatterns, finalSets, info

    def generateTimeStamps(self, node):
        finalTs = node.timeStamps
        return finalTs

    def removeNode(self, nodeValue):
        """
            removing the node from tree

            :param nodeValue : it represents the node in tree
            :type nodeValue : node
        """
        for i in self.summaries[nodeValue]:
            i.parent.timeStamps = i.parent.timeStamps + i.timeStamps
            del i.parent.children[nodeValue]

    def getTimeStamps(self, alpha):
        """
        Returns the timeStamps of a node

        Parameters
        ----------
        alpha: node of tree

        Returns
        -------
        timeStamps of a node

        """
        temporary = []
        for i in self.summaries[alpha]:
            temporary += i.timeStamps
        return temporary

    def getPeriodicSupport(self, timeStamps):
        """
                    calculates the support and periodicity with list of timestamps

                    :param timeStamps : timestamps of a pattern
                    :type timeStamps : list


                            """
        timeStamps = list(set(timeStamps))
        timeStamps.sort()
        per = 0
        sup = 0
        for i in range(len(timeStamps) - 1):
            j = i + 1
            if abs(timeStamps[j] - timeStamps[i]) <= period:
                per += 1
            sup += 1
        return per

    def conditionalTransactions(self, conditionalPatterns, conditionalTimeStamps):
        """ It generates the conditional patterns with periodic frequent items

                :param conditionalPatterns : conditional_patterns generated from condition_pattern method for
                                        respective node
                :type conditionalPatterns : list
                :param conditionalTimeStamps : represents the timestamps of conditional patterns of a node
                :type conditionalTimeStamps : list
        """
        global periodicSupport, period
        patterns = []
        timeStamps = []
        data1 = {}
        for i in range(len(conditionalPatterns)):
            for j in conditionalPatterns[i]:
                if j in data1:
                    data1[j] = data1[j] + conditionalTimeStamps[i]
                else:
                    data1[j] = conditionalTimeStamps[i]
        updatedDictionary = {}
        for m in data1:
            updatedDictionary[m] = self.getPeriodicSupport(data1[m])
        updatedDictionary = {k: v for k, v in updatedDictionary.items() if v >= periodicSupport}
        count = 0
        for p in conditionalPatterns:
            p1 = [v for v in p if v in updatedDictionary]
            trans = sorted(p1, key=lambda x: (updatedDictionary.get(x), -x), reverse=True)
            if len(trans) > 0:
                patterns.append(trans)
                timeStamps.append(conditionalTimeStamps[count])
            count += 1
        return patterns, timeStamps, updatedDictionary

    def generatePatterns(self, prefix):
        """generates the patterns

                :param prefix : forms the combination of items
                :type prefix : list
                        """
        for i in sorted(self.summaries, key=lambda x: (self.info.get(x), -x)):
            pattern = prefix[:]
            pattern.append(i)
            yield pattern, self.info[i]
            patterns, timeStamps, info = self.getConditionalPatterns(i)
            conditionalTree = Tree()
            conditionalTree.info = info.copy()
            for pat in range(len(patterns)):
                conditionalTree.addTransaction(patterns[pat], timeStamps[pat])
            if len(patterns) > 0:
                for q in conditionalTree.generatePatterns(pattern):
                    yield q
            self.removeNode(i)


class ThreePGrowth(partialPeriodicPatterns):

    """ 3pgrowth is fundamental approach to mine the partial periodic patterns in temporal database.

        Reference : Discovering Partial Periodic Itemsets in Temporal Databases,SSDBM '17: Proceedings of the 29th International Conference on Scientific and Statistical Database ManagementJune 2017
        Article No.: 30 Pages 1–6https://doi.org/10.1145/3085504.3085535

        Parameters:
        ----------
        self.iFile : file
            Name of the Input file or path of the input file
        self. oFile : file
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
        self.memoryUSS : float
            To store the total amount of USS memory consumed by the program
        self.memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        self.startTime:float
            To record the start time of the mining process
        self.endTime:float
            To record the completion time of the mining process
        Database : list
            To store the transactions of a database in list
        mapSupport : Dictionary
            To maintain the information of item and their frequency
        lno : int
            it represents the total no of transactions
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
        partialPeriodicOneItem()
            Extracts the one-frequent patterns from transactions
        updateTransactions()
            updates the transactions by removing the aperiodic items and sort the transactions with items
            by decreaing support
        buildTree()
            constrcuts the main tree by setting the root node as null
        startMine()
            main program to mine the partial periodic patterns

        Format: python3 threePGrowth.py <inputFile> <outputFile> <periodicSupport> <period>

        Examples: python3 threePGrowth.py sampleDB.txt patterns.txt 10.0 2.0   (periodicSupport and period will be considered in percentage of database transactions)

                  python3 threePGrowth.py sampleDB.txt patterns.txt 10 2     (periodicSupprot and period will be considered in count)
        
        Sample run of the importing code:
        -----------
        from PAMI.periodicFrequentPattern.basic import threePGrowth as alg

        obj = alg.ThreePGrowth(iFile, periodicSupport, period)

        obj.startMine()

        partialPeriodicPatterns = obj.getPatterns()

        print("Total number of partial periodic Patterns:", len(partialPeriodicPatterns))

        obj.storePatternsInFile(oFile)

        Df = obj.getPatternInDf()

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
    periodicSupport = float()
    period = float()
    startTime = float()
    endTime = float()
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
            Storing the complete transactions of the database/input file in a database variable


            """
        try:
            with open(self.iFile, 'r', encoding='utf-8') as f:
                for line in f:
                    li1 = [i.strip() for i in line.split(self.sep)]
                    self.Database.append(li1)
                    self.lno += 1
        except IOError:
            print("File Not Found")

    def partialPeriodicOneItem(self):
        """
                    calculates the support of each item in the dataset and assign the ranks to the items
                    by decreasing support and returns the frequent items list

                    """
        data = {}
        for tr in self.Database:
            for i in range(1, len(tr)):
                if tr[i] not in data:
                    data[tr[i]] = [0, int(tr[0]), 1]
                else:
                    lp = int(tr[0]) - data[tr[i]][1]
                    if lp <= self.period:
                        data[tr[i]][0] += 1
                    data[tr[i]][1] = int(tr[0])
                    data[tr[i]][2] += 1
        data = {k: v[0] for k, v in data.items() if v[0] >= self.periodicSupport}
        pfList = [k for k, v in sorted(data.items(), key=lambda x: x[1], reverse=True)]
        self.rank = dict([(index, item) for (item, index) in enumerate(pfList)])
        return data, pfList

    def updateTransactions(self, dict1):
        """remove the items which are not frequent from transactions and updates the transactions with rank of items

                    :param dict1 : frequent items with support
                    :type dict1 : dictionary
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

    def buildTree(self, data, info):
        """it takes the transactions and support of each item and construct the main tree with setting root
                            node as null

                                :param data : it represents the one transactions in database
                                :type data : list
                                :param info : it represents the support of each item
                                :type info : dictionary
                                """
        rootNode = Tree()
        rootNode.info = info.copy()
        for i in range(len(data)):
            set1 = []
            set1.append(data[i][0])
            rootNode.addTransaction(data[i][1:], set1)
        return rootNode

    def savePeriodic(self, itemset):
        """
        To convert the pattern with its original item name
        :param itemset: partial periodic pattern
        :return: pattern with original item name
        """
        temp = str()
        for i in itemset:
            temp = temp + self.rankdup[i] + " "
        return temp

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
        """
                   Main method where the patterns are mined by constructing tree.

               """
        global periodicSupport, period, lno
        self.startTime = time.time()
        if self.iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self.periodicSupport is None:
            raise Exception("Please enter the Minimum Support")
        self.creatingItemSets()
        self.periodicSupport = self.convert(self.periodicSupport)
        self.period = self.convert(self.period)
        periodicSupport, period, lno = self.periodicSupport, self.period, len(self.Database)
        if self.periodicSupport > len(self.Database):
            raise Exception("Please enter the minSup in range between 0 to 1")
        generatedItems, pfList = self.partialPeriodicOneItem()
        updatedTransactions = self.updateTransactions(generatedItems)
        for x, y in self.rank.items():
            self.rankdup[y] = x
        info = {self.rank[k]: v for k, v in generatedItems.items()}
        Tree = self.buildTree(updatedTransactions, info)
        patterns = Tree.generatePatterns([])
        for i in patterns:
            s = self.savePeriodic(i[0])
            self.finalPatterns[s] = i[1]
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Partial Periodic Patterns were generated successfully using 3PGrowth algorithm ")

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
            data.append([a, b[0], b[1]])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
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
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:
            ap = ThreePGrowth(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 5:
            ap = ThreePGrowth(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Partial Periodic Patterns:", len(Patterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
