from pyspark.context import SparkContext
from pyspark.conf import SparkConf
# import abstract as _ab
from PAMI.frequentPattern.pyspark import abstract as _ab
from collections import OrderedDict

class Node:
    """
    Attribute
    ---------
        item : int
            Storing item of a node
        count : int
            To maintain the support count of node
        children : dict
            To maintain the children of node
        prefix : list
            To maintain the prefix of node
    """

    def __init__(self,item,count,children):
        self.item = item
        self.count = count
        self.children = children
        self.prefix = []


class Tree:
    """
    Attribute
    ---------
        root : Node
            The first node of the tree set to Null
        nodeLink : dict
            Store nodes that have the same item

    Methods
    -------
        createTree(transaction,count)
            Create tree from transaction and count
        linkNode(node)
            Add nodes that have the same item to self.nodeLink
        createCPB(item)
            Create conditional pattern base of item
        mergeTree(tree,fpList)
            Merge tree
        createTransactions(fpList)
            Create transactions from yourself(tree)
        getPattern(item,suffixItem,minSup)
            Get frequent patterns about suffixItem from yourself
        mining(minSup,isResponsible = lambda x:True)
            Do pattern mining on your own
    """


    def __init__(self):
        self.root = Node(None,0,{})
        self.nodeLink = OrderedDict()


    def createTree(self,transaction,count):
        """
        Adding transaction into tree
        :param transaction: list
        :param count: int
        :return: Tree
        """
        current = self.root
        for item in transaction:
            if item not in current.children:
                current.children[item] = Node(item,count,{})
                current.children[item].prefix = transaction[0:transaction.index(item)]
                self.linkNode(current.children[item])
            else:
                current.children[item].count += count
            current = current.children[item]
        return self


    def linkNode(self, node):
        """
        Adding node to nodeLinks
        :param node: Node
        :return:
        """
        if node.item in self.nodeLink:
            self.nodeLink[node.item].append(node)
        else:
            self.nodeLink[node.item] = []
            self.nodeLink[node.item].append(node)


    def createCPB(self,item):
        """
        Create conditional pattern base based on item
        :param item: int
        :return: Tree
        """
        pTree = Tree()
        for node in self.nodeLink[item]:
            pTree.createTree(node.prefix,node.count)
        return pTree


    def mergeTree(self,tree,fpList):
        """
        Merging tree to self
        :param tree: Tree
        :param fpList: list
        :return: Tree
        """
        transactions = tree.createTransactions(fpList)
        for transaction in transactions:
            self.createTree(transaction,1)
        return self


    def createTransactions(self,fpList):
        """
        Creating transactions from self
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


    def getPattern(self,item,suffixItem,minSup):
        """
        Getting frequent patterns about suffixItem
        :param item: int
        :param suffixItem: tuple
        :param minSup: int
        :return: list
        """
        pTree = self.createCPB(item)
        frequentItems = {}
        frequentPatterns = []
        for i in pTree.nodeLink.keys():
            frequentItems[i] = 0
            for node in pTree.nodeLink[i]:
                frequentItems[i] += node.count
        frequentItems = {key: value for key, value in frequentItems.items() if value >= minSup}
        for i in frequentItems:
            pattern = list(suffixItem)
            pattern.append(i)
            frequentPatterns.append((tuple(pattern),frequentItems[i]))
            frequentPatterns.extend(pTree.getPattern(i, tuple(pattern), minSup))
        return frequentPatterns


    def mining(self,minSup,isResponsible = lambda x:True):
        """
        Mining self and return frequent patterns
        :param minSup: int
        :param isResponsible: function
        :return: list
        """
        frequentPatterns = []
        flist = sorted([item for item in self.nodeLink.keys()])
        for item in reversed(flist):
            if isResponsible(item):
                frequentPatterns.extend(self.getPattern(item,tuple([item]),minSup))
        return frequentPatterns



class parallelFPGrowth(_ab._frequentPatterns):
    """
        Attributes
        ----------
            minSup : float
                The user can specify minSup either in count or proportion of database size.
            iFile : file
                Input file name or path of the input file.
            oFile : file
                Name of the output file or the path of the output file.
            sep : str
                This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
                However, the users can override their default separator.
            startTime:float
                To record the start time of the mining process
            endTime:float
                To record the completion time of the mining process
            memoryUSS : float
                To store the total amou.nt of USS memory consumed by the program
            memoryRSS : float
                To store the total amount of RSS memory consumed by the program
            finalPatterns : dict
                it represents to store the all frequent patterns
            numWorkers: int
                The number of workers
                This value means the number of cores which are used.

        Methods
        -------
            startMine()
                Mining process will start from this function
            getPatterns()
                Complete set of patterns will be retrieved with this function
            savePatterns(outFile)
                Complete set of frequent patterns will be loaded in to a output file
            getPatternsAsDataFrame()
                Complete set of frequent patterns will be loaded in to a dataframe
            getMemoryUSS()
                Total amount of USS memory consumed by the mining process will be retrieved from this function
            getMemoryRSS()
                Total amount of RSS memory consumed by the mining process will be retrieved from this function
            getRuntime()
                Total amount of runtime taken by the mining process will be retrieved from this function
            getFrequentItems(data)
                Get frequent items that length is 1.
            genCondTransaction(data, rank)
                Generating conditional transactions for distributed pattern mining
            getPartitionId(item)
                Get partition id of item
                FPTree is created on each workers based on partition id.
            getAllFrequentPatterns(data, fpList)
                Get all frequent patterns
       
       Executing the code on terminal:
       -------------------------------
            
            Format:
            ------
            
                python3 parallelFPGrowth.py <inputFile> <outputFile> <minSup> <numWorkers>
            
            Examples:
            ---------
                python3 parallelFPGrowth.py sampleDB.txt patterns.txt 10.0 3   (minSup will be considered in times of minSup and count of database transactions)
            
                python3 parallelFPGrowth.py sampleDB.txt patterns.txt 10 3    (minSup will be considered in support count or frequency)
       
       Sample run of the importing code:
       ---------------------------------
            
            import PAMI.frequentPattern.pyspark.parallelFPGrowth as alg
            
            obj = alg.parallelFPGrowth(iFile, minSup, numWorkers)
            
            obj.startMine()
            
            frequentPatterns = obj.getPatterns()
            
            print("Total number of Frequent Patterns:", len(frequentPatterns))
            
            obj.savePatterns(oFile)
            
            Df = obj.getPatternInDataFrame()
            
            memUSS = obj.getMemoryUSS()
            
            print("Total Memory in USS:", memUSS)
            
            memRSS = obj.getMemoryRSS()
            
            print("Total Memory in RSS", memRSS)
            
            run = obj.getRuntime()
            
            print("Total ExecutionTime in seconds:", run)
        
        Credits:
        --------
            The complete program was written by Yudai Masu under the supervision of Professor Rage Uday Kiran.
            
    """

    _minSup = float()
    _numWorkers = int()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()

    def __init__(self, iFile, minSup, numWorkers, sep = '\t'):
        super().__init__(iFile, float(minSup), int(numWorkers), sep)

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

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def savePatterns(self, outFile):
        """
        Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            if type(x) == tuple:
                pattern = ""
                for item in x:
                    pattern = pattern + str(item) + " "
                s1 = pattern + ":" + str(y)
            else:
                s1 = str(x) + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def _getFrequentItems(self, data):
        """
        Get the frequent items that length is 1
        :param data: RDD
        :return: dict
        """
        flatData = data.flatMap(lambda x: [(y, 1) for y in x])
        oneFrequentItems = dict(flatData.reduceByKey(lambda x, y: x + y)
                                .filter(lambda c: c[1] >= self._minSup).collect())
        return oneFrequentItems

    def _getAllFrequentPatterns(self, data, FPList):
        """
        Get all frequent patterns
        :param data: RDD
        :param fpList: list
        :return: list
        """
        rank = dict([(index, item) for (item, index) in enumerate(FPList)])
        newFPList = list(rank.values())
        workByPartition = data.flatMap(lambda transaction: self._genCondTransaction(transaction, rank))
        emptyTree = Tree()
        forest = workByPartition.aggregateByKey(emptyTree, lambda tree, transaction: tree.createTree(transaction, 1),
                                                lambda tree1, tree2: tree1.mergeTree(tree2, newFPList))
        frequentItemsets = forest.flatMap(lambda tree_tuple:
                                          tree_tuple[1].mining(self._minSup,
                                                               lambda x: self._getPartitionId(x) == tree_tuple[0]))
        I = frequentItemsets.map(lambda ranks_count: ([FPList[int(z)] for z in ranks_count[0]], ranks_count[1]))
        return I

    def _getPartitionId(self, item):
        """
        Get partition id of item
        :param item: int
        :return: int
        """
        return item % self._numWorkers

    def _genCondTransaction(self, transaction, rank):
        """
        Generate conditional transactions from transaction
        :param transaction : list
        :param rank: dict
        :return: list
        """
        filtered = [rank[int(x)] for x in transaction if int(x) in rank.keys()]
        filtered = sorted(filtered)
        condTransaction = {}
        for i in range(len(filtered) - 1, -1, -1):
            item = filtered[i]
            partition = self._getPartitionId(item)
            if partition not in condTransaction:
                condTransaction[partition] = filtered[:i + 1]
        return [x for x in condTransaction.items()]
    
    def _convert(self, dataLength, value):
        """
        To convert the user specified minSup value
        :param value: user specified minSup value
        :return: converted type
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (dataLength * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (dataLength * value)
            else:
                value = int(value)
        return value

    def startMine(self):
        """
        Main program to start the operation
        """
        self._startTime = _ab._time.time()
        conf = SparkConf().setAppName("FPGrowth").setMaster("local[*]")
        sc = SparkContext(conf=conf)
        #sc.addFile("fpTree.py")

        data = sc.textFile(self._iFile, self._numWorkers).map(lambda x: [int(y) for y in x.strip().split(self._sep)])\
            .persist()
        self._minSup = self._convert(data.count(), self._minSup)
        frequentItems = self._getFrequentItems(data)
        self._finalPatterns.update(frequentItems)

        FPList = [x for (x, y) in sorted(frequentItems.items(), key=lambda x: -x[1])]
        frequentPatterns = self._getAllFrequentPatterns(data, FPList)

        for v in list(frequentPatterns.collect()):
            temp = (tuple(v[0]), v[1])
            self._finalPatterns[temp[0]] = temp[1]

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Parallel FPGrowth algorithm")
        sc.stop()




if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = PFPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = PFPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        _finalPatterns = _ap.getPatterns()
        print("Total number of Frequent Patterns:", len(_finalPatterns))
        _ap.savePatterns(_ab._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
