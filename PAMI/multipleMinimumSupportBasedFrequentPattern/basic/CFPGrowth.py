# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#
#     from PAMI.multipleMinimumSupportBasedFrequentPattern.basic import CFPGrowth as alg
#
#     obj = alg.CFPGrowth(iFile, mIS)
#
#     obj.startMine()
#
#     frequentPatterns = obj.getPatterns()
#
#     print("Total number of Frequent Patterns:", len(frequentPatterns))
#
#     obj.save(oFile)
#
#     Df = obj.getPatternInDataFrame()
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


from PAMI.multipleMinimumSupportBasedFrequentPattern.basic import abstract as _fp

_fp._sys.setrecursionlimit(20000)
_MIS = {}

class _Node:
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


class _Tree:
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
        self.root = _Node(None, {})
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
                newNode = _Node(transaction[i], {})
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

    @staticmethod
    def getConditionalTransactions(ConditionalPatterns, conditionalFreq):
        """
        To calculate the frequency of items in conditional patterns and sorting the patterns
        Parameters
        ----------
        ConditionalPatterns: paths of a node
        conditionalFreq: frequency of each item in the path

        Returns
        -------
            conditional patterns and frequency of each item in transactions
        """
        global _minSup
        pat = []
        freq = []
        data1 = {}
        for i in range(len(ConditionalPatterns)):
            for j in ConditionalPatterns[i]:
                if j in data1:
                    data1[j] += conditionalFreq[i]
                else:
                    data1[j] = conditionalFreq[i]
        #up_dict = {k: v for k, v in data1.items() if v >= _minSup}
        up_dict = data1.copy()
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
        global _MIS
        for i in sorted(self.summaries, key=lambda x: (self.info.get(x), -x)):
            pattern = prefix[:]
            pattern.append(i)
            sup = []
            for j in pattern:
                sup.append(_MIS[j])
            if self.info[i] >= min(sup):
                yield pattern, self.info[i]
            patterns, freq, info = self.getFinalConditionalPatterns(i)
            conditionalTree = _Tree()
            conditionalTree.info = info.copy()
            for pat in range(len(patterns)):
                conditionalTree.addTransaction(patterns[pat], freq[pat])
            if len(patterns) > 0:
                for q in conditionalTree.generatePatterns(pattern):
                    yield q


class CFPGrowth(_fp._frequentPatterns):
    """
    Description:
    ------------------

       CFPGrowth is one of the fundamental algorithm to discover frequent patterns based on multiple minimum support in a transactional database.

    Reference :
    ---------
        Ya-Han Hu and Yen-Liang Chen. 2006. Mining association rules with multiple minimum supports: a new mining algorithm and a support tuning mechanism.
        Decis. Support Syst. 42, 1 (October 2006), 1–24. https://doi.org/10.1016/j.dss.2004.09.007


    Attributes :
    ----------
        iFile : file
            Input file name or path of the input file
        MIS: file or dictionary
            Multiple minimum supports of all items in the database
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default separator is tab space or \t.
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
    ----------------
        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        save(oFile)
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
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
    ----------------------------------
        Format:
        -------
            >>> python3 CFPGrowth.py <inputFile> <outputFile>

        Examples:
        ---------
            >>> python3 CFPGrowth.py sampleDB.txt patterns.txt MISFile.txt

            >>> python3 CFPGrowth.py sampleDB.txt patterns.txt MISFile.txt

            >>> python3 CFPGrowth.py sampleTDB.txt output.txt sampleN.txt MIS ',' (it will consider "," as a separator)


    Sample run of the importing code:
    --------------------------------------


        from PAMI.multipleMinimumSupportBasedFrequentPattern.basic import CFPGrowth as alg

        obj = alg.CFPGrowth(iFile, mIS)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.save(oFile)

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

    __startTime = float()
    __endTime = float()
    _MIS = str
    __finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    __memoryUSS = float()
    __memoryRSS = float()
    __Database = []
    __mapSupport = {}
    __lno = 0
    __tree = _Tree()
    __rank = {}
    __rankDup = {}

    def __init__(self, iFile, MIS, sep='\t'):
        super().__init__(iFile, MIS, sep)

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

            # print(self.Database)
        if isinstance(self._iFile, str):
            if _fp._validators.url(self._iFile):
                data = _fp._urlopen(self._iFile)
                for line in data:
                    line = line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self.__Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            temp = [i.rstrip() for i in line.split('\t')]
                            temp = [x for x in temp if x]
                            # print(temp)
                            self.__Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def _getMISValues(self):
        """
            Storing the Minimum supports given by the user for each item in the database


        """
        self._MISValues = {}
        if isinstance(self._MIS, _fp._pd.DataFrame):
            items, MIS = [], []
            if self._MIS.empty:
                print("its empty..")
            i = self._MIS.columns.values.tolist()
            if 'items' in i:
                items = self._MIS['items'].tolist()
            if 'MIS' in i:
                MIS = self._MIS['MIS'].tolist()
            for i in range(len(items)):
                self._MISValues[items[i]] = MIS[i]

        if isinstance(self._MIS, str):
            if _fp._validators.url(self._MIS):
                data = _fp._urlopen(self._MIS)
                for line in data:
                    line = line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._MISValues[temp[0]] = int(temp[1])
            else:
                try:
                    with open(self._MIS, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._MISValues[temp[0]] = int(temp[1])
                    print(len(self._MISValues))
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
                value = (len(self.__Database) * value)
            else:
                value = int(value)
        return value

    def __frequentOneItem(self):
        """
        Generating One frequent items sets

        """
        self.__mapSupport = {}
        for tr in self.__Database:
            for i in range(len(tr)):
                if tr[i] not in self.__mapSupport:
                    self.__mapSupport[tr[i]] = 1
                else:
                    self.__mapSupport[tr[i]] += 1
        # for x, y in self.__mapSupport.items():
        #     print(x, y)
        self.__mapSupport = {k: v for k, v in self.__mapSupport.items() if v >= min(self._MISValues.values())}
        # for x, y in self.__mapSupport.items():
        #     print(x, y)
        genList = [k for k, v in sorted(self.__mapSupport.items(), key=lambda x: x[1], reverse=True)]
        self.__rank = dict([(index, item) for (item, index) in enumerate(genList)])
        return genList

    def __updateTransactions(self, itemSet):
        """
        Updates the items in transactions with rank of items according to their support

        :Example: oneLength = {'a':7, 'b': 5, 'c':'4', 'd':3}
                    rank = {'a':0, 'b':1, 'c':2, 'd':3}

        Parameters
        ----------
        itemSet: list of one-frequent items

        -------

        """
        list1 = []
        for tr in self.__Database:
            list2 = []
            for i in range(len(tr)):
                if tr[i] in itemSet:
                    list2.append(self.__rank[tr[i]])
            if len(list2) >= 1:
                list2.sort()
                list1.append(list2)
        return list1

    @staticmethod
    def __buildTree(transactions, info):
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
        rootNode = _Tree()
        rootNode.info = info.copy()
        for i in range(len(transactions)):
            rootNode.addTransaction(transactions[i], 1)
        return rootNode

    def __savePeriodic(self, itemSet):
        """
        The duplication items and their ranks
        Parameters:
        ----------
            itemSet: frequent itemSet that generated

        Returns:
        -------
            patterns with original item names.

        """
        temp = str()
        for i in itemSet:
            temp = temp + self.__rankDup[i] + "\t"
        return temp

    def startMine(self):
        """
            main program to start the operation

        """
        global _MIS
        self.__startTime = _fp._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        self.__creatingItemSets()
        self._getMISValues()
        #MIS = self._MISValues
        itemSet = self.__frequentOneItem()
        updatedTransactions = self.__updateTransactions(itemSet)
        for x, y in self.__rank.items():
            _MIS[y] = self._MISValues[x]
            self.__rankDup[y] = x
        info = {self.__rank[k]: v for k, v in self.__mapSupport.items()}
        __Tree = self.__buildTree(updatedTransactions, info)
        patterns = __Tree.generatePatterns([])
        self.__finalPatterns = {}
        for k in patterns:
            s = self.__savePeriodic(k[0])
            self.__finalPatterns[str(s)] = k[1]
        print("Frequent patterns were generated successfully using CFPGrowth algorithm")
        self.__endTime = _fp._time.time()
        self.__memoryUSS = float()
        self.__memoryRSS = float()
        process = _fp._psutil.Process(_fp._os.getpid())
        self.__memoryUSS = process.memory_full_info().uss
        self.__memoryRSS = process.memory_info().rss

    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process

        :rtype: float
        """

        return self.__memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process

        :rtype: float
        """

        return self.__memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process


        :return: returning total amount of runtime taken by the mining process

        :rtype: float
        """

        return self.__endTime - self.__startTime

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame
        """

        dataframe = {}
        data = []
        for a, b in self.__finalPatterns.items():
            data.append([a.replace('\t', ' '), b])
            dataframe = _fp._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataframe

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self.__finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns

        :rtype: dict
        """
        return self.__finalPatterns

    def printResults(self):
        print("Total number of  Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_fp._sys.argv) == 4 or len(_fp._sys.argv) == 5:
        if len(_fp._sys.argv) == 5:
            _ap = CFPGrowth(_fp._sys.argv[1], _fp._sys.argv[3], _fp._sys.argv[4])
        if len(_fp._sys.argv) == 4:
            _ap = CFPGrowth(_fp._sys.argv[1], _fp._sys.argv[3])
        _ap.startMine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_fp._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        _ap = CFPGrowth('/Users/Likhitha/Downloads/Transactional_T10I4D100K-3.csv', '/Users/Likhitha/Downloads/MIS_T10I4D100K_.csv', '\t')
        _ap.startMine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save('/Users/Likhitha/Downloads/CFPGrowth_output.txt')
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
        print("Error! The number of input parameters do not match the total number of parameters provided")
