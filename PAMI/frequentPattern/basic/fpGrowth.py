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

from PAMI.frequentPattern.basic.abstract import *
import sys

minSup = str()


class Node:
    """
        A class used to represent the node of frequentPatternTree

    Attributes:
    ----------
        itemId: int
            storing item of a node
        counter: int
            To maintain the support of node
        parent: node
            To maintain the parent of node
        children: list
            To maintain the children of node

    Methods:
    -------

        addChild(node)
            Updates the nodes children list and parent for the given node

    """

    def __init__(self, item, children):
        self.itemId = item
        self.counter = 1
        self.parent = None
        self.children = children

    def addChild(self, node):
        """
            Retrieving the child from the tree

            :param node: Children node
            :type node: Node
            :return: Updates the children nodes and parent nodes

        """
        self.children[node.itemId] = node
        node.parent = self


class Tree:
    """
    A class used to represent the frequentPatternGrowth tree structure

    Attributes:
    ----------
        root : Node
            The first node of the tree set to Null.
        summaries : dictionary
            Stores the nodes itemId which shares same itemId
        info : dictionary
            frequency of items in the transactions

    Methods:
    -------
        addTransaction(transaction, freq)
            adding items of  transactions into the tree as nodes and freq is the count of nodes
        getFinalConditionalPatterns(node)
            getting the conditional patterns from fp-tree for a node
        getConditionalPatterns(patterns, frequencies)
            sort the patterns by removing the items with lower minSup
        generatePatterns(prefix)
            generating the patterns from fp-tree
    """

    def __init__(self):
        self.root = Node(None, {})
        self.summaries = {}
        self.info = {}

    def addTransaction(self, transaction, count):
        """adding transaction into tree

        :param transaction: it represents the one transactions in database

        :type transaction: list

        :param count: frequency of item

        :type count: int
        """

        # This method takes transaction as input and returns the tree
        currentNode = self.root
        for i in range(len(transaction)):
            if transaction[i] not in currentNode.children:
                newNode = Node(transaction[i], {})
                newNode.freq = count
                currentNode.addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]
                currentNode.freq += count

    def getFinalConditionalPatterns(self, alpha):
        """
        generates the conditional patterns for a node

        Parameters:
        ----------
            alpha: node to generate conditional patterns

        Returns
        -------
            returns conditional patterns, frequency of each item in conditional patterns

        """
        finalPatterns = []
        finalFreq = []
        for i in self.summaries[alpha]:
            set1 = i.freq
            set2 = []
            while i.parent.itemId is not None:
                set2.append(i.parent.itemId)
                i = i.parent
            if len(set2) > 0:
                set2.reverse()
                finalPatterns.append(set2)
                finalFreq.append(set1)
        finalPatterns, finalFreq, info = self.getConditionalTransactions(finalPatterns, finalFreq)
        return finalPatterns, finalFreq, info

    def getConditionalTransactions(self, ConditionalPatterns, conditionalFreq):
        """
        To calculate the frequency of items in conditional patterns and sorting the patterns
        Parameters
        ----------
        ConditionalPatterns: paths of a node
        conditionalFreq: frequency of each item in the path

        Returns
        -------
            conditional patterns and frequncy of each item in transactions
        """
        global minSup
        pat = []
        freq = []
        data1 = {}
        for i in range(len(ConditionalPatterns)):
            for j in ConditionalPatterns[i]:
                if j in data1:
                    data1[j] += conditionalFreq[i]
                else:
                    data1[j] = conditionalFreq[i]
        up_dict = {k: v for k, v in data1.items() if v >= minSup}
        count = 0
        for p in ConditionalPatterns:
            p1 = [v for v in p if v in up_dict]
            trans = sorted(p1, key=lambda x: (up_dict.get(x), -x), reverse=True)
            if len(trans) > 0:
                pat.append(trans)
                freq.append(conditionalFreq[count])
            count += 1
        return pat, freq, up_dict

    def generatePatterns(self, prefix):
        """
        To generate the frequent patterns
        Parameters
        ----------
        prefix: an empty list

        Returns
        -------
        Frequent patterns that are extracted from fp-tree

        """
        for i in sorted(self.summaries, key=lambda x: (self.info.get(x), -x)):
            pattern = prefix[:]
            pattern.append(i)
            yield pattern, self.info[i]
            patterns, freq, info = self.getFinalConditionalPatterns(i)
            conditionalTree = Tree()
            conditionalTree.info = info.copy()
            for pat in range(len(patterns)):
                conditionalTree.addTransaction(patterns[pat], freq[pat])
            if len(patterns) > 0:
                for q in conditionalTree.generatePatterns(pattern):
                    yield q


class fpGrowth(frequentPatterns):
    """
       fpGrowth is one of the fundamental algorithm to discover frequent patterns in a transactional database.
       It stores the database in compressed fp-tree decreasing the memory usage and extracts the
       patterns from tree.It employs employs downward closure property to  reduce the search space effectively.

    Reference :
    ---------
           Han, J., Pei, J., Yin, Y. et al. Mining Frequent Patterns without Candidate Generation: A Frequent-Pattern
           Tree Approach. Data  Mining and Knowledge Discovery 8, 53–87 (2004). https://doi.org/10.1023

    Attributes :
    ----------
        iFile : file
            Input file name or path of the input file
        minSup: float or int or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
            However, the users can override their default separator.
        oFile : file
            Name of the output file or the path of the output file
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
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

    Methods :
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
            Extracts the one-frequent patterns from transactions
            
    Executing the code on terminal:
    -------
        Format:
        -------
            python3 fpGrowth.py <inputFile> <outputFile> <minSup>

        Examples:
        ---------
            python3 fpGrowth.py sampleDB.txt patterns.txt 10.0   (minSup will be considered in percentage of database transactions)

            python3 fpGrowth.py sampleDB.txt patterns.txt 10     (minSup will be considered in support count or frequency)


    Sample run of the importing code:
    -----------


        from PAMI.frequentPattern.basic import fpGrowth as alg

        obj = alg.fpGrowth(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.storePatternsInFile(oFile)

        Df = obj.getPatternInDataFrame()

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
    finalPatterns = {}
    iFile = " "
    oFile = " "
    sep = " "
    memoryUSS = float()
    memoryRSS = float()
    Database = []
    mapSupport = {}
    lno = 0
    tree = Tree()
    rank = {}
    rankDup = {}

    def __init__(self,iFile,minSup,sep='\t'):
        super().__init__(iFile,minSup,sep)



    def creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable


        """
        try:
            with open(self.iFile, 'r', encoding='utf-8') as f:
                for line in f:
                    self.lno += 1
                    li1 = [i.rstrip() for i in line.split(self.sep)]
                    self.Database.append(li1)
        except IOError:
            print("File Not Found")

    def convert(self, value):
        """
        to convert the type of user specified minSup value

        :param value: user specified minSup value

        :return: converted type
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


    def frequentOneItem(self):
        """
        Generating One frequent items sets

        """
        for tr in self.Database:
            for i in range(0, len(tr)):
                if tr[i] not in self.mapSupport:
                    self.mapSupport[tr[i]] = 1
                else:
                    self.mapSupport[tr[i]] += 1
        self.mapSupport = {k: v for k, v in self.mapSupport.items() if v >= self.minSup}
        genList = [k for k, v in sorted(self.mapSupport.items(), key=lambda x: x[1], reverse=True)]
        self.rank = dict([(index, item) for (item, index) in enumerate(genList)])
        return genList

    def updateTransactions(self, itemSet):
        """
        Updates the items in transactions with rank of items according to their support

        Parameters
        ----------
        itemSet: list of one-frequent items

        Returns: Updated transactions with rank assigning to each item and deletes the unfequent items
        -------

        """
        list1 = []
        for tr in self.Database:
            list2 = []
            for i in range(len(tr)):
                if tr[i] in itemSet:
                    list2.append(self.rank[tr[i]])
            if len(list2) >= 1:
                list2.sort()
                list1.append(list2)
        return list1

    def buildTree(self, transactions, info):
        """
        Builds the tree with updated transactions
        Parameters:
        ----------
            transactions: updated transactions
            info: support details of each item in transactions

        Returns:
        -------
            transactions compressed in fp-tree

        """
        rootNode = Tree()
        rootNode.info = info.copy()
        for i in range(len(transactions)):
            rootNode.addTransaction(transactions[i], 1)
        return rootNode

    def savePeriodic(self, itemSet):
        """
        The duplication items and their ranks
        Parameters:
        ----------
            itemSet: frequent itemset that generated

        Returns:
        -------
            patterns with original item names.

        """
        temp = str()
        for i in itemSet:
            temp = temp + self.rankDup[i] + " "
        return temp

    def startMine(self):
        """
            main program to start the operation

        """
        global minSup
        self.startTime = time.time()
        if self.iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self.minSup is None:
            raise Exception("Please enter the Minimum Support")
        self.creatingItemSets()
        self.minSup = self.convert(self.minSup)
        minSup = self.minSup
        itemSet = self.frequentOneItem()
        updatedTransactions = self.updateTransactions(itemSet)
        for x, y in self.rank.items():
            self.rankDup[y] = x
        info = {self.rank[k]: v for k, v in self.mapSupport.items()}
        Tree = self.buildTree(updatedTransactions, info)
        patterns = Tree.generatePatterns([])
        for k in patterns:
            s = self.savePeriodic(k[0])
            self.finalPatterns[str(s)] = k[1]
        print("Frequent patterns were generated successfully using frequentPatternGrowth algorithm")
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
            data.append([a, b])
            dataframe = pd.DataFrame(data, columns=['Patterns', 'Support'])
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
    if len(sys.argv) == 4 or len(sys.argv) == 5:
        if len(sys.argv) == 5:
            ap = fpGrowth(sys.argv[1], sys.argv[3], sys.argv[4])
        if len(sys.argv) == 4:
            ap = fpGrowth(sys.argv[1], sys.argv[3])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Frequent Patterns:", len(Patterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
