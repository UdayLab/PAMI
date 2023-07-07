__copyright__ = """
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
"""


import abstract as _fp

_fp._sys.setrecursionlimit(20000)

class _Node:
    """
    This class represents a node in the FP-tree

    Attributes:
    ----------
        itemId : str
            The item name of the node

        counter : int
            The support count of the item

        parent : Node
            The parent node of the current node

        children : dict
            The dictionary of children of the current node

        tail : list
            The tail of the current node denoting the support count in each batch

        next : Node
            The next node of the current node

        
        Methods:
        -------

        addChild(node) 
            Adds a child node to the current node

        shiftTail()
            Shifts the tail of the current node during the removal of the batch

        addSupportCount(count)
            Adds the support count to the current node

    """
    def __init__(self, item, supportCount):
        self.itemId = item
        self.counter = supportCount
        self.parent = None
        self.children = {}
        self.tail = None
        self.next = None

    def addChild(self, node):
        """
            Adds a child node to the current node

            :param node: The child node to be added
            :type node: Node
        """
        self.children[node.itemId] = node
        node.parent = self

    def shiftTail(self):
        """
            Shifts the tail of the current node during the removal of the batch
        """
        self.tail.pop(0)
        self.tail.append(0)

    def addSupportCount(self, count):
        """
            Adds the support count to the current node

            :param count: The support count to be added
            :type count: int
        """
        self.counter += count

class _HeaderTable:

    """
        This class represents the header table of the FP-tree

        Attributes:
        ----------
            table : dict
                The dictionary of items and their support count with node pointer

            orderedItems : list
                The list of items in the header table in the order of their support count

        Methods:
        -------
            updateSupportCount(item, count, node)
                Updates the support count with the node pointer of the item in the header table

            addSupportCount(item, count)
                Adds the support count of the item in the header table

            removeSupportCount(item, count)
                Removes the support count of the item in the header table

            itemOrdering()
                Orders the items in the header table in the order of their support count

            orderTransaction(transaction)
                Orders the items in the transaction in the order of their support count

    """

    def __init__(self):
        self.table = {}
        self.orderedItems = []

    def updateSupportCount(self, item, count, node):
        """
            Updates the support count with the node pointer of the item in the header table

            :param item: The item whose support count is to be updated
            :type item: str
            :param count: The support count to be added
            :type count: int
            :param node: The node pointer to be added
            :type node: Node
        """
        if item in self.table:
            self.table[item][0] += count
            tempNode = self.table[item][1]
            while tempNode.next != None:
                tempNode = tempNode.next
            tempNode.next = node
        else:
            self.table[item] = [count, node]

        self.itemOrdering()

    def addSupportCount(self, item, count):
        """
            Adds the support count of the item in the header table

            :param item: The item whose support count is to be added
            :type item: str
            :param count: The support count to be added
            :type count: int
        """
        if item in self.table:
            self.table[item][0] += count
    
    def removeSupportCount(self, item, count):
        """
            Removes the support count of the item in the header table

            :param item: The item whose support count is to be removed
            :type item: str
            :param count: The support count to be removed
            :type count: int
        """
        if item in self.table:
            self.table[item][0] -= count
            if(self.table[item][0] == 0):
                del self.table[item]
    
    def itemOrdering(self):
        """
            Orders the items in the header table in the order of their support count
        """
        self.orderedItems = list(sorted(self.table.keys(), key=lambda x: self.table[x][0]))

    def orderTransaction(self, transaction):
        """
            Orders the items in the transaction in the order of their support count

            :param transaction: The transaction to be ordered
            :type transaction: list

            :return: The ordered transaction
            :rtype: list
        """
        return sorted(transaction, key=lambda x: self.table[x][0] if x in self.table else 1, reverse=True)

class _CPSTree:
    """
        Class representing the CPSTree

        Attributes:
        ----------
            root : Node
                The root node of the CPSTree

            headerTable : HeaderTable
                The header table of the CPSTree

            windowSize : int
                The window size used for analyzing the data stream

            paneSize : int
                The pane size used in each window for analyzing the data stream

            curBatchIndex : int
                The current batch index of the data stream

        Methods:
        -------

            addTransaction(transaction, restructuring = False, supportValue = None, tailNode = None)
                Adds the transaction to the CPSTree

            updateBranch(node, count)
                Updates the branch of the node with the support count useful while removing the oldest batch

            removeBatch()
                Removes the oldest batch from the CPSTree

    """

    def __init__(self, windowSize, paneSize):
        self.root = _Node(None, 0)
        self.headerTable = _HeaderTable()
        self.windowSize = windowSize
        self.paneSize = paneSize
        self.curBatchIndex = 0

    def addTransaction(self, transaction, restructuring = False, supportValue = None, tailNode = None):
        """
            Adds the transaction to the CPSTree

            :param transaction: The transaction to be added
            :type transaction: list
            :param restructuring: Flag to indicate whether the transaction is added during restructuring
            :type restructuring: bool
            :param supportValue: The support count of the items in the transaction useful during restructuring
            :type supportValue: dict
            :param tailNode: The tail node of the transaction useful during restructuring
            :type tailNode: Node
        """
        curNode = self.root

        if restructuring is False:
            transaction = self.headerTable.orderTransaction(transaction)

        for item in transaction:
            if item in curNode.children:

                if supportValue is None:
                    curNode.children[item].addSupportCount(1)
                    self.headerTable.addSupportCount(item, 1)
                else:
                    curNode.children[item].addSupportCount(supportValue[item])
                    self.headerTable.addSupportCount(item, supportValue[item])
                
                curNode = curNode.children[item]
            else:

                if supportValue is None:
                    newNode = _Node(item, 1)
                    self.headerTable.updateSupportCount(item, 1, newNode)

                else:
                    newNode = _Node(item, supportValue[item])
                    self.headerTable.updateSupportCount(item, supportValue[item], newNode)

                curNode.addChild(newNode)
                newNode.parent = curNode
                curNode = newNode

        if curNode.tail is None:
            curNode.tail = [0] * self.windowSize

        if restructuring is False:
            if supportValue is None:
                curNode.tail[self.curBatchIndex] = 1
            else:
                curNode.tail[self.curBatchIndex] = supportValue[item]


        if tailNode is not None and curNode.tail is None:
            curNode.tail = tailNode.copy()

        if tailNode is not None and curNode.tail is not None:
            for i in range(len(tailNode)):
                curNode.tail[i] += tailNode[i]
            

    def updateBranch(self, curNode, curSupport):
        """
            Updates the branch of the node with the support count useful while removing the oldest batch

            :param curNode: The node whose branch is to be updated
            :type curNode: Node
            :param curSupport: The support count of the node to be updated
            :type curSupport: int
        """
        if(curNode.itemId is None):
            return

        curNode.addSupportCount(-curSupport)
        parentNode = curNode.parent

        if(curNode.counter == 0):
            if(self.headerTable.table[curNode.itemId][1] == curNode):
                self.headerTable.table[curNode.itemId][1] = curNode.next

            else:
                prev = self.headerTable.table[curNode.itemId][1]
                while(prev is not None and prev.next != curNode):
                    prev = prev.next

                prev.next = curNode.next

            del curNode.parent.children[curNode.itemId]

        self.headerTable.removeSupportCount(curNode.itemId, curSupport)
        self.updateBranch(parentNode, curSupport)


    def removeBatch(self):
        """
            Removes the oldest batch from the CPSTree by recursively iterating over the tree
        """
        root = self.root
        curItems = list(self.headerTable.table.keys())
        for item in curItems:
            curNode = self.headerTable.table[item][1]
            while curNode is not None:
                if curNode.tail is not None:
                    curSupport = curNode.tail[0]

                    if(curSupport != 0):
                        self.updateBranch(curNode, curSupport)
                    curNode.shiftTail()

                curNode = curNode.next


class FrequentPatternStreamMining(_fp._frequentPatternsStream):
    """
        Description:
        -------------

            Sliding window-based frequent pattern mining over data streams is one of basic data stream mining tasks.
            The goal is to find all frequent patterns in a sliding window over a data stream. It stores the complete
            database in form of FP-Tree but adjusts the tree structure based on the upcoming transactions and removes
            the oldest batch from the tree. This creates a sliding window over the data stream. The adjusting FP-tree
            structure is called CPSTree. It is a modified version of the existing FP-Tree structure.

        Reference : 
        ------------
            Syed Khairuzzaman Tanbeer and Chowdhury Farhan Ahmed and Byeong-Soo Jeong and Young-Koo Lee : Sliding 
            window-based frequent pattern mining over data streams. Information Sciences Vol 179, 3843 - 3865, 2009.
            https://doi.org/10.1016/j.ins.2009.07.012

        Attributes : 
        ------------
            iFile : str
                The input file name or path

            minSup : float or int or str
                The user can specify minSup either in count or proportion of database size.
                If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
                Otherwise, it will be treated as float.
                Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float

            sep : str
                This variable is used to distinguish items from one another in a transaction. The default separator is tab space or \t.
                However, the users can override their default separator.

            windowSize : int
                The size of the sliding window. It specifies the number of panes to be considered for mining frequent patterns.

            paneSize : int
                The size of the pane. It specifies the number of transactions to be considered for mining frequent patterns in each pane.

            oFile : str
                The output file name or path

            startTime : float
                The start time of the mining process

            endTime : float
                The end time of the mining process

            finalPatterns : dict
                The dictionary containing the mined frequent patterns from each window.

            memoryUSS : float
                The memory usage of the program in User Space

            memoryRSS : float
                The memory usage of the program in Resident Set Size

            Database : list
                The list containing the complete transactions of the database/input file

            tree : CPSTree
                The CPSTree structure used to mine frequent patterns from the data stream


        Methods :
        ---------
            creatingItemSets()
                Loads the complete transactions of the database/input file in a database variable.

            convert(value):
                Converts the minSup from different formats in absolute count.

            getRestructuredTree(tree):
                Restructures the tree by removing the oldest batch from the tree and updating the new batch in the tree.

            getBranches(root, restructuredTree, originalTree, curBranch):
                Gets the branches of the tree and updates it to the restructured tree.

            printTree():
                Prints the CPSTree structure.

            createPrefixBranch(root):
                Creates the branches of the prefix tree.

            createConditionalTree(root, transactions, minSup):
                Creates the conditional tree from the prefix tree by removing items with support less than minSup.

            itemSetGeneration(root, minSup, candidatePattern, curItemset):
                Generates the frequent itemsets by creating a conditional tree and mining patterns from it.

            
            getMemoryRSS():
                Gets the memory usage of the program in Resident Set Size.

            getMemoryUSS():
                Gets the memory usage of the program in User Space.

            getPatterns():
                Gets the mined frequent patterns from each window.

            getPatternsAsDataFrame():
                Gets the mined frequent patterns from each window as a pandas dataframe.

            getRuntime():
                Gets the runtime of the program.

            printResults():
                Prints the overall stasticts of the mining process.

            save():
                Saves the mined frequent patterns in a file.

            startMine():
                Starts the mining process by loading the input database and creating CPS Tree over each window.

        
        Executing the code on terminal:
        -------------------------------

            Format:
            --------
                python3 CPSTree.py <inputFile> <outputFile> <minSup> <windowSize> <paneSize> <separator>

            Example:
            ---------

                python3 CPSTree.py retail.txt output.txt 0.5 100 1000 ',' (Here minimum support is 50% of the database size

        Credits:
        --------

            The code is written by Vipul Chhabra under the supervision of Prof. Rage Uday Kiran.

    """

    __startTime = float()
    __endTime = float()
    _minSup = str()
    __finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    __memoryUSS = float()
    __memoryRSS = float()
    __Database = []
    __tree = None
    __windowSize = 0
    __paneSize = 0

    def __init__(self, iFile, oFile, minSup, windowSize, paneSize, sep='\t'):
        super().__init__(iFile, minSup, windowSize, paneSize, sep)
        self._oFile = oFile

    def __creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable
        """
        self.__Database = []
        if isinstance(self._iFile, _fp._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self.__Database = self._iFile['Transactions'].tolist()

        if isinstance(self._iFile, str):
            if _fp._validators.url(self._iFile):
                data = _fp._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self.__Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self.__Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def __convert(self, value):
        """
            to convert the type of user specified minSup value

            :param value: user specified minSup value

            :return: converted type
        """

        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self.__Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self.__paneSize * self.__windowSize * value)
            else:
                value = int(value)
        return value
    
    def getRestructuredTree(self,tree):
        """
            Restructures the tree by removing the oldest batch from the tree and updating the new batch in the tree.

            :param tree: Existing CPSTree object to be restructured
            
            :return: restructuredTree: CPSTree object after performing branch sorting
        """
        restructuredTree = _CPSTree(tree.windowSize, tree.paneSize)
        self.getBranches(tree.root, restructuredTree, tree)
        restructuredTree.curBatchIndex = tree.curBatchIndex
        return restructuredTree

    def getBranches(self,root, restructuredTree, origTree, curBranch = []):
        """
            Gets the branches of the tree and updates it to the restructured tree.

            :param root: CPSTreeNode object for the root of the original tree
            :param restructuredTree: CPSTree object
            :param origTree: CPSTree object for the original tree
            :param curBranch: list of CPSTreeNode objects that needs to be sorted
        """

        if(root is None):
            return

        for childItem in root.children:
            updatedBranch = curBranch.copy()
            updatedBranch.append(root.children[childItem])
            self.getBranches(root.children[childItem], restructuredTree, origTree, updatedBranch)

        if(len(root.children) == 0 or root.tail is not None):
            finalBranch = curBranch.copy()
            supportCount = {}

            for node in finalBranch:
                supportCount[node.itemId] = sum(finalBranch[-1].tail)

            curTrans = [node.itemId for node in finalBranch]
            curTrans = origTree.headerTable.orderTransaction(curTrans)
            restructuredTree.addTransaction(curTrans, restructuring=True, supportValue=supportCount, tailNode = finalBranch[-1].tail)
    
    def printTree(self, root, level = 0):
        """
            Prints the tree in a readable format.

            :param root: CPSTreeNode object for the root of the tree
            :param level: Current level of the root node
        """

        print('  ' * level, level, root.itemId, root.counter, root.parent.itemId if root.parent else None )
        
        if(len(root.children) == 0 or root.tail != None):
            print('  ' * (level + 1), level + 1, root.tail)
        
        for child in root.children.values():
            self.printTree(child, level + 1)

    def createPrefixBranch(self, root):
        """
            Creates the prefix branch for the given node.

            :param root: CPSTreeNode object for the root of the tree

            :return: prefixBranch: list of CPSTreeNode objects for the prefix branch
        """

        stack = []
        while(root is not None):
            stack.append(root)
            root = root.parent

        chosenItemset = stack[0]
        lastSupport = chosenItemset.counter
        return stack, lastSupport

    def createConditionalTree(self, root, transactions, minSupport):
        """
            Creates the conditional tree for the given node.

            :param root: CPSTreeNode object for the root of the tree for which conditional tree needs to be created
            :param transactions: list of transactions which needs to be added to the conditional tree
            :param minSupport: minimum support value for the conditional tree

            :return: conditionalTree: CPSTree object for the conditional tree
        """    
    
        for transaction in transactions:
            for item in transaction["transaction"]:
                if root.headerTable.table[item][0] < minSupport:
                    transaction["transaction"].remove(item)

        conditionalTree = _CPSTree(1, 1)

        for transaction in transactions:
            if(len(transaction["transaction"]) > 0):
                conditionalTree.addTransaction(transaction["transaction"], restructuring=True, supportValue=transaction["support"])

        return conditionalTree      

    def itemSetGeneration(self, root, minSupport, candidatePattern, curItem = []):

        """
            Generates the candidate patterns for the given CPS Tree.

            :param root: CPSTreeNode object for the root of the tree for which itemset needs to be generated
            :param minSupport: minimum support value for the conditional tree
            :param candidatePattern: list of candidate patterns generated
            :param curItem: list of CPSTreeNode objects for the current itemset

            :return: candidatePattern: list of candidate patterns
        """
        if(root is None or root.root is None):
            return

        for item in root.headerTable.orderedItems:
            if root.headerTable.table[item][0] >= minSupport:
                prefixBranches = []
                tempNode = root.headerTable.table[item][1]

                while(tempNode is not None):
                    curPrefixBranch, netSupport = self.createPrefixBranch(tempNode)
                    tempBranch = [node.itemId for node in curPrefixBranch]
                    supportCount = {}
                    for each_item_node in curPrefixBranch:
                        supportCount[each_item_node.itemId] = netSupport

                    prefixBranches.append([tempBranch, supportCount])
                    tempNode = tempNode.next

                prefixTree = _CPSTree(1, 1)
                completeTransactions = []

                for branch in prefixBranches:
                    transaction = []
                    for nodeIndex in range(len(branch[0])-2,0,-1):
                        transaction.append(branch[0][nodeIndex])
                    
                    prefixTree.addTransaction(transaction, restructuring=True, supportValue=branch[1])
                    completeTransactions.append({"transaction": transaction, "support": branch[1]})

                conditionalTree = self.createConditionalTree(prefixTree, completeTransactions, minSupport)
                newItemset = curItem.copy()
                newItemset.append(item)

                if(len(newItemset) not in candidatePattern):
                    candidatePattern[len(newItemset)] = [{"pattern" : newItemset, "support" : root.headerTable.table[item][0]}]

                else:
                    candidatePattern[len(newItemset)].append({"pattern" : newItemset, "support" : root.headerTable.table[item][0]})

                if(len(conditionalTree.headerTable.table) > 0):
                    self.itemSetGeneration(conditionalTree, minSupport, candidatePattern, newItemset)


    def getMemoryRSS(self):
        """
            Total amount of RSS memory consumed by the mining process will be retrieved from this function

            :return: returning RSS memory consumed by the mining process

            :rtype: float
        """
        return self.__memoryRSS

    def getMemoryUSS(self):
        """
            Total amount of USS memory consumed by the mining process will be retrieved from this function

            :return: returning USS memory consumed by the mining process

            :rtype: float
        """
        return self.__memoryUSS

    def getPatterns(self):
        """
            Returns the frequent patterns generated by the mining process over the complete datastream.

            :return: returning frequent patterns generated by the mining process

            :rtype: dict
        """
        return self.__finalPatterns

    def getPatternsAsDataFrame(self):
        """
            Stores the final patterns generated by the mining process in a dataframe.

            :return: returning dataframe containing the final patterns generated by the mining process

            :rtype: pandas.DataFrame
        """

        dataframe = {}
        data = []
        for x, y in self.__finalPatterns.items():
            for pattern in y:
                patternString = ' '.join(pattern[0])
                data.append([x[0], x[1], patternString, pattern[1]])
        dataframe = _fp._pd.DataFrame(data, columns=['Window Start Index', 'Window End Index', 'Pattern', 'Support'])
        return dataframe

    def getRuntime(self):
        """
            Total amount of time taken by the mining process will be retrieved from this function

            :return: returning time taken by the mining process

            :rtype: float
        """

        return self.__endTime - self.__startTime

    def printResults(self):
        """
            Prints the stats of the mining process

        """
        print("Total number of Windows Processed:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


    def save(self):
        """
        Complete set of frequent patterns will be loaded in to a output file
        """

        writer = open(self._oFile, 'w+')
        for x, y in self.__finalPatterns.items():
            writer.write("Window Start Index : %s , End Index : %s \n" % (x[0], x[1]))
            for pattern in y:
                patternString = '\t'.join(pattern[0])
                patternString += '\t' + ":" + '\t' + str(pattern[1])
                writer.write("%s \n" % patternString)

    def startMine(self):
        """
            This function will start the mining process
        """
        global _minSup
        self.__startTime = _fp._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._minSup is None:
            raise Exception("Please enter the Minimum Support")
        if self._windowSize is None:
            raise Exception("Please enter the Window Size")
        if self._paneSize is None:
            raise Exception("Please enter the Pane Size")
        self.__windowSize = int(self._windowSize)
        self.__paneSize = int(self._paneSize)

        self.__creatingItemSets()
        self._minSup = self.__convert(self._minSup)
        self.__tree = _CPSTree(self.__windowSize, self.__paneSize)

        for i in range(0, self.__windowSize):
            self.__tree.curBatchIndex = i
            for j in range(0, self.__paneSize):
                    self.__tree.addTransaction(self.__Database[i * self.__paneSize + j])
            self.__tree = self.getRestructuredTree(self.__tree)


        startIndex = 0
        endIndex = self.__windowSize * self.__paneSize

        while endIndex <= len(self.__Database):
            print("Start Index: ", startIndex, "End Index: ", endIndex)
            self.printTree(self.__tree.root)

            patterns = {}
            self.itemSetGeneration(self.__tree, self._minSup, patterns)

            for patternLength in patterns:
                for pattern in patterns[patternLength]:

                    if((startIndex, endIndex) not in self.__finalPatterns):
                        self.__finalPatterns[(startIndex, endIndex)] = []
                    self.__finalPatterns[(startIndex, endIndex)].append((pattern["pattern"], pattern["support"]))

            if(endIndex == len(self.__Database)):
                break

            self.__tree.removeBatch()

            for i in range(0, self.__paneSize):
                    self.__tree.addTransaction(self.__Database[endIndex + i])

            self.__tree = self.getRestructuredTree(self.__tree)
            startIndex += self.__paneSize
            endIndex += self.__paneSize



        self.__endTime = _fp._time.time()
        self.__memoryUSS = float()
        self.__memoryRSS = float()
        process = _fp._psutil.Process(_fp._os.getpid())
        self.__memoryUSS = process.memory_full_info().uss
        self.__memoryRSS = process.memory_info().rss



if __name__ == "__main__":
    _ap = str()
    if len(_fp._sys.argv) == 6 or len(_fp._sys.argv) == 7:
        if len(_fp._sys.argv) == 7:
            _ap = FrequentPatternStreamMining(_fp._sys.argv[1], _fp._sys.argv[2], _fp._sys.argv[3], _fp._sys.argv[4], _fp._sys.argv[5], _fp._sys.argv[6])
        if len(_fp._sys.argv) == 6:
            _ap = FrequentPatternStreamMining(_fp._sys.argv[1], _fp._sys.argv[2], _fp._sys.argv[3], _fp._sys.argv[4], _fp._sys.argv[5])
        _ap.startMine()
        print("Total number of Frequent Patterns:", len( _ap.getPatterns()))

        print(_ap.getPatternsAsDataFrame().to_csv("result.csv", index = False, sep='\t'))
        _ap.save()
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")