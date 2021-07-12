import sys

import pandas as pd

from abstract import *

class Node:
    """
    A class used to represent the node of frequentPatternTree

        ...

        Attributes
        ----------
        itemId : int
            storing item of a node
        counter : int
            To maintain the support of node
        parent : node
            To maintain the parent of every node
        child : list
            To maintain the children of node
        nodeLink : node
            Points to the node with same itemId

        Methods
        -------

        getChild(itemName)
            returns the node with same itemName from frequentPatternTree
    """

    def __init__(self):
        self.itemId = -1
        self.counter = 1
        self.parent = None
        self.child = []
        self.nodeLink = None

    def getChild(self, id1):
        for i in self.child:
            if i.itemId == id1:
                return i
        return []


class Tree:
    """
        A class used to represent the frequentPatternGrowth tree structure

        ...

        Attributes
        ----------
        headerList : list
            storing the list of items in tree sorted in ascending of their supports
        mapItemNodes : dictionary
            storing the nodes with same item name
        mapItemLastNodes : dictionary
            representing the map that indicates the last node for each item
        root : Node
            representing the root Node in a tree


        Methods
        -------
        createHeaderList(items,minSup)
            takes items only which are greater than minSup and sort the items in ascending order
        addTransaction(transaction)
            creating transaction as a branch in frequentPatternTree
        fixNodeLinks(item,newNode)
            To create the link for nodes with same item
        printTree(Node)
            gives the details of node in frequentPatternGrowth tree
        addPrefixPath(prefix,port,minSup)
           It takes the items in prefix pattern whose support is >=minSup and construct a subtree
    """

    def __init__(self):
        """

        """
        self.headerList = []
        self.mapItemNodes = {}
        self.mapItemLastNodes = {}
        self.root = Node()

    def addTransaction(self, transaction):
        """adding transaction into tree

        :param transaction : it represents the one transactions in database
        :type transaction : list
        """

        # This method taken a transaction as input and returns the tree
        current = self.root
        for i in transaction:
            child = current.getChild(i)
            if child == []:
                newNode = Node()
                newNode.itemId = i
                newNode.parent = current
                current.child.append(newNode)
                self.fixNodeLinks(i, newNode)
                current = newNode
            else:
                child.counter += 1
                current = child

    def fixNodeLinks(self, item, newNode):
        """Fixing node link for the newNode that inserted into frequentPatternTree

        :param item: it represents the item of newNode
        :type item : int
        :param newNode : it represents the newNode that inserted in frequentPatternTree
        :type newNode : Node

        """
        if item in self.mapItemLastNodes.keys():
            lastNode = self.mapItemLastNodes[item]
            lastNode.nodeLink = newNode
        self.mapItemLastNodes[item] = newNode
        if item not in self.mapItemNodes.keys():
            self.mapItemNodes[item] = newNode

    def printTree(self, root):
        """Print the details of Node in frequentPatternTree

        :param root: it represents the Node in frequentPatternTree
        :type root: Node

        This method is to find the details of parent,children,support of Node
        """

        # this method is used print the details of tree
        if root.child == []:
            return
        else:
            for i in root.child:
                print(i.itemId, i.counter, i.parent.itemId)
                self.printTree(i)

    def createHeaderList(self, mapSupport, minsup):
        """To create the headerList

        :param mapSupport : it represents the items with their supports
        :type mapSupport : dictionary
        :param minsup : it represents the minSup
        :param minsup : float
        """
        # the frequentPatternTree always maintains the header table to start the mining from leaf nodes
        t1 = []
        for x, y in mapSupport.items():
            if y >= minsup:
                t1.append(x)
        itemSetBuffer = [k for k, v in sorted(mapSupport.items(), key=lambda x: x[1], reverse=True)]
        self.headerList = [i for i in t1 if i in itemSetBuffer]

    def addPrefixPath(self, prefix, mapSupportBeta, minsup):
        """To construct the conditional tree with prefix paths of a node in frequentPatternTree

        :param prefix : it represents the prefix items of a Node
        :type prefix : list
        :param mapSupportBeta : it represents the items with their supports
        :param mapSupportBeta : dictionary
        :param minsup : to check the item meets with minSup
        :param minsup : float
        """
        # this method is used to add prefix paths in conditional trees of frequentPatternTree
        pathCount = prefix[0].counter
        current = self.root
        prefix.reverse()
        for i in range(0, len(prefix) - 1):
            pathItem = prefix[i]
            if mapSupportBeta.get(pathItem.itemId) >= minsup:
                child = current.getChild(pathItem.itemId)
                if child == []:
                    newNode = Node()
                    newNode.itemId = pathItem.itemId
                    newNode.parent = current
                    newNode.counter = pathCount
                    current.child.append(newNode)
                    current = newNode
                    self.fixNodeLinks(pathItem.itemId, newNode)
                else:
                    child.counter += pathCount
                    current = child


class cpgrowth(frequentPatterns):
    """
        cpgrowth is one of the fundamental algorithm to discover Corelated frequent patterns in a transactional database.
        it is based on traditional Fpgrowth Algorithm,This algorithm uses breadth-first search technique to find the 
        corelated Frequent patterns in transactional database.
    
        Parameters
        ----------
        self.iFile : file
            Name of the Input file to mine complete set of frequent patterns
       self. oFile : file
            Name of the output file to store complete set of frequent patterns
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        minSup : int
            The user given minSup
        cratio: float
        	The user given Ratio
        Database : list
            To store the transactions of a database in list
        mapSupport : Dictionary
            To maintain the information of item and their frequency
        lno : int
            it represents the total no of transactions
        tree : class
            it represents the Tree class
        itemSetCount : int
            it represents the total no of patterns
        finalPatterns : dict
            it represents to store the patterns
        itemSetBuffer : list
            it represents the store the items in mining
        maxPatternLength : int
           it represents the constraint for pattern length

        Methods
        -------
        startMine()
            Mining process will start from here
        getFrequentPatterns()
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
        check(line)
            To check the delimiter used in the user input file
        creatingItemSets(fileName)
            Scans the dataset or dataframes and stores in list format
        frequentOneItem()
            Extracts the one-frequent patterns from transactions
        saveAllCombination(tempBuffer,s,position,prefix,prefixLength)
            Forms all the combinations between prefix and tempBuffer lists with support(s)
        saveItemSet(pattern,support)
            Stores all the frequent patterns with their respective support
        frequentPatternGrowthGenerate(frequentPatternTree,prefix,port)
            Mining the frequent patterns by forming conditional frequentPatternTrees to particular prefix item.
            mapSupport represents the 1-length items with their respective support

        Executing the code on terminal
        ------------------------------

        Format: python3 apriori.py <inputFile> <outputFile> <minSup> <ratio>

        Examples: python3 cpgrowth.py inp.txt output.txt 4.0 0.3   (minSup will be considered in percentage of database transactions)
                  python3 cpgrowth.py  patterns.txt 4  0.3   (minSup will be considered in support count or frequency)

        """

    startTime = float()
    endTime = float()
    minSup = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    memoryUSS = float()
    memoryRSS = float()
    Database = []
    mapSupport = {}
    lno = 0
    tree = Tree()
    itemSetBuffer = None
    fpNodeTempBuffer = []
    cratio = 0.0
    itemSetCount = 0
    maxPatternLength = 1000

    def __init__(self, iFile, minsup, ratio):
        super().__init__(iFile,minsup,ratio)

    def creatingItemSets(self, iFileName):
        """
            Storing the complete transactions of the database/input file in a database variable

            :param iFileName: user given input file/input file path
            :type iFileName: str


            """
        self.Database = []
        with open(iFileName, 'r', encoding='utf-8') as f:
            for line in f:
                line.strip()
                if self.lno == 0:
                    self.lno += 1
                    # delimiter = self.findDelimiter([*line])
                    li1 = [i.rstrip() for i in line.split("	")]
                    self.Database.append(li1)
                else:
                    self.lno += 1
                    li1 = [i.rstrip() for i in line.split("	")]
                    self.Database.append(li1)

    def getRatio(self, prefix, prefixLength, s):
        """
    		A FUnction to get itemset Ration
    		:param prefix:the path
    		:type prefix: list
    		:param prefixLength: length
    		:type prefixLength:int
    		:s :current ratio
    		:type s:float
    		:return: ratio of prefix
    		:rtype:float
    	"""
        maximums = 0
        for ele in range(prefixLength):
            i=prefix[ele]
            if maximums < self.mapSupport.get(i):
                maximums = self.mapSupport.get(i)
        return s / maximums

    def frequentOneItem(self):
        """Generating One frequent items sets

        """
        for i in self.Database:
            for j in i:
                if j not in self.mapSupport:
                    self.mapSupport[j] = 1
                else:
                    self.mapSupport[j] += 1

    def saveItemSet(self, prefix, prefixLength, support):
        """To save the frequent patterns mined form frequentPatternTree

        :param prefix: the frequent pattern
        :type prefix: list
        :param prefixLength : the length of a frequent pattern
        :type prefixLength : int
        :param support: the support of a pattern
        :type support :  int
        :The frequent patterns are update into global variable finalPatterns
        """
        ratio = self.getRatio(prefix, prefixLength, support)
        if ratio < self.minRatio:
            return
        l = []
        for i in range(prefixLength):
            l.append(prefix[i])
        self.itemSetCount += 1
        self.finalPatterns[tuple(l)] = str(support)+" : "+str(ratio)
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

    def saveAllCombinations(self, tempBuffer, s, position, prefix, prefixLength):
        """Generating all the combinations for items in single branch in frequentPatternTree

        :param tempBuffer: items in a list
        :type tempBuffer: list
        :param s : support at leaf node of a branch
        :param position : the length of a tempBuffer
        :type position : int
        :param prefix : it represents the list of leaf node
        :type prefix : list
        :param prefixLength : the length of prefix
        :type prefixLength :int
        
        """
        max1 = 1 << position
        for i in range(1, max1):
            newPrefixLength = prefixLength
            for j in range(position):
                isSet = i & (1 << j)
                if isSet > 0:
                    prefix.insert(newPrefixLength, tempBuffer[j].itemId)
                    newPrefixLength += 1
            self.saveItemSet(prefix, newPrefixLength, s)

    def frequentPatternGrowthGenerate(self, frequentPatternTree, prefix, prefixLength, mapSupport):
        """

        Mining the fp tree

        :param frequentPatternTree: it represents the frequentPatternTree
        :type frequentPatternTree: class Tree
        :param prefix : it represents a empty list and store the patterns that are mined
        :type prefix : list
        :param param prefixLength : the length of prefix
        :type prefixLength :int
        :param mapSupport : it represents the support of item
        :type mapSupport : dictionary

        """

        singlePath = True
        position = 0
        s = 0
        if len(frequentPatternTree.root.child) > 1:
            singlePath = False
        else:
            currentNode = frequentPatternTree.root.child[0]
            while True:
                if len(currentNode.child) > 1:
                    singlePath = False
                    break
                self.fpNodeTempBuffer.insert(position, currentNode)
                s = currentNode.counter
                position += 1
                if len(currentNode.child) == 0:
                    break
                currentNode = currentNode.child[0]
        if singlePath is True:
            self.saveAllCombinations(self.fpNodeTempBuffer, s, position, prefix, prefixLength)
        else:
            for i in reversed(frequentPatternTree.headerList):
                item = i
                support = mapSupport[i]
                betaSupport = support
                prefix.insert(prefixLength, item)
                self.saveItemSet(prefix, prefixLength + 1, betaSupport)
                if prefixLength + 1 < self.maxPatternLength:
                    prefixPaths = []
                    path = frequentPatternTree.mapItemNodes.get(item)
                    mapSupportBeta = {}
                    while path is not None:
                        if path.parent.itemId != -1:
                            prefixPath = []
                            prefixPath.append(path)
                            pathCount = path.counter
                            parent1 = path.parent
                            while parent1.itemId != -1:
                                prefixPath.append(parent1)
                                if mapSupportBeta.get(parent1.itemId) is None:
                                    mapSupportBeta[parent1.itemId] = pathCount
                                else:
                                    mapSupportBeta[parent1.itemId] = mapSupportBeta[parent1.itemId] + pathCount
                                parent1 = parent1.parent
                            prefixPaths.append(prefixPath)
                        path = path.nodeLink
                    treeBeta = Tree()
                    for k in prefixPaths:
                        treeBeta.addPrefixPath(k, mapSupportBeta, self.minSup)
                    if len(treeBeta.root.child) > 0:
                        treeBeta.createHeaderList(mapSupportBeta, self.minSup)
                        self.frequentPatternGrowthGenerate(treeBeta, prefix, prefixLength + 1, mapSupportBeta)
    

    def startMine(self):
        """
        main program to start the operation

        """

        self.startTime = time.time()
        if self.iFile == None:
            raise Exception("Please enter the file path or file name:")
        iFileName = self.iFile
        cratio = self.minRatio
        self.creatingItemSets(iFileName)
        self.minSup=self.convert(self.minSup)
        print("minSup: ", self.minSup)
        self.frequentOneItem()
        self.mapSupport = {k: v for k, v in self.mapSupport.items() if v >= self.minSup}
        itemSetBuffer = [k for k, v in sorted(self.mapSupport.items(), key=lambda x: x[1], reverse=True)]
        for i in self.Database:
            transaction = []
            for j in i:
                if j in itemSetBuffer:
                    transaction.append(j)
            transaction.sort(key=lambda val: self.mapSupport[val], reverse=True)
            self.tree.addTransaction(transaction)
        self.tree.createHeaderList(self.mapSupport, self.minSup)
        if len(self.tree.headerList) > 0:
            self.itemSetBuffer = []
            self.frequentPatternGrowthGenerate(self.tree, self.itemSetBuffer, 0, self.mapSupport)
        print("Corelated Frequent patterns were generated successfully using CorelatedePatternGrowth algorithm")
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss

    def getMemoryUSS(self):
        """
        Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self.memoryUSS

    def getMemoryRSS(self):
        """
        Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self.memoryRSS

    def getRuntime(self):
        """
        Calculating the total amount of runtime taken by the mining process


        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self.endTime - self.startTime

    def getPatternsInDataFrame(self):
        """

        Storing final frequent patterns in a dataframe

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
        """
        Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            pat = ""
            for i in x:
                pat += str(i) + " "
            patternsAndSupport = pat + ": " + str(y)
            writer.write("%s \n" % patternsAndSupport)

    def getFrequentPatterns(self):
        """
        Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Error! Input Argument miss-match, please enter 4 arguments only")
    else:
        ap = cpgrowth(sys.argv[1], sys.argv[3], float(sys.argv[4]))
        ap.startMine()
        frequentPatterns = ap.getFrequentPatterns()
        print("Total number of Frequent Patterns:", len(frequentPatterns))
        output = sys.argv[2]
        ap.storePatternsInFile(output)
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in seconds:", run)
