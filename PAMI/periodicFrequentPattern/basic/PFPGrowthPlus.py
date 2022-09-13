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


from PAMI.periodicFrequentPattern.basic import abstract as _ab

_maxPer = float()
_minSup = float()
_lno = int()


class _Node(object):
    """
    A class used to represent the node of frequentPatternTree

    ...
    Attributes:
    ----------
        item : int
            storing item of a node
        timeStamps : list
            To maintain the timestamps of transaction at the end of the branch
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
        self.children[node.item] = node
        node.parent = self


class _Tree(object):
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
                starts from the root node of the tree and mines the periodic-frequent patterns

        """

    def __init__(self):
        self.root = _Node(None, {})
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
                newNode = _Node(transaction[i], {})
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
        """generates all the conditional patterns of respective node

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

    @staticmethod
    def generateTimeStamps(node):
        finalTimeStamps = node.timeStamps
        return finalTimeStamps

    def removeNode(self, nodeValue):
        """removing the node from tree

                        :param nodeValue : it represents the node in tree
                        :type nodeValue : node
                        """
        for i in self.summaries[nodeValue]:
            i.parent.timeStamps = i.parent.timeStamps + i.timeStamps
            del i.parent.children[nodeValue]

    def getTimeStamps(self, alpha):
        temporary = []
        for i in self.summaries[alpha]:
            temporary += i.timeStamps
        return temporary

    @staticmethod
    def getSupportAndPeriod(timeStamps):
        """
                   calculates the support and periodicity with list of timestamps

                   :param timeStamps : timestamps of a pattern
                   :type timeStamps : list


                           """
        global _maxPer, _lno
        timeStamps.sort()
        cur = 0
        per = 0
        sup = 0
        for j in range(len(timeStamps)):
            per = max(per, timeStamps[j] - cur)
            if per > _maxPer:
                return [0, 0]
            cur = timeStamps[j]
            sup += 1
        per = max(per, _lno - cur)
        return [sup, per]

    def conditionalTransactions(self, conditionalPatterns, conditionalTimeStamps):
        """ It generates the conditional patterns with periodic frequent items

                :param conditionalPatterns : conditionalPatterns generated from conditionalPattern method for
                                    respective node
                :type conditionalPatterns : list
                :param conditionalTimeStamps : represents the timestamps of conditional patterns of a node
                :type conditionalTimeStamps : list
                """
        global _maxPer, _minSup
        pat = []
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
            updatedDictionary[m] = self.getSupportAndPeriod(data1[m])
        updatedDictionary = {k: v for k, v in updatedDictionary.items() if v[0] >= _minSup and v[1] <= _maxPer}
        count = 0
        for p in conditionalPatterns:
            p1 = [v for v in p if v in updatedDictionary]
            trans = sorted(p1, key=lambda x: (updatedDictionary.get(x)[0], -x), reverse=True)
            if len(trans) > 0:
                pat.append(trans)
                timeStamps.append(conditionalTimeStamps[count])
            count += 1
        return pat, timeStamps, updatedDictionary

    def generatePatterns(self, prefix):
        """generates the patterns

                :param prefix : forms the combination of items
                :type prefix : list
                """
        for i in sorted(self.summaries, key=lambda x: (self.info.get(x)[0], -x)):
            pattern = prefix[:]
            pattern.append(i)
            yield pattern, self.info[i]
            patterns, timeStamps, info = self.getConditionalPatterns(i)
            conditionalTree = _Tree()
            conditionalTree.info = info.copy()
            for pat in range(len(patterns)):
                conditionalTree.addTransaction(patterns[pat], timeStamps[pat])
            if len(patterns) > 0:
                for q in conditionalTree.generatePatterns(pattern):
                    yield q
            self.removeNode(i)


class PFPGrowthPlus(_ab._periodicFrequentPatterns):
    """ PFPGrowthPlus is fundamental and improved version of PFPGrowth algorithm to discover periodic-frequent patterns in temporal database.
        It uses greedy approach to discover effectively

        Reference :
        --------
        R. UdayKiran, MasaruKitsuregawa, and P. KrishnaReddyd, "Efficient discovery of periodic-frequent patterns in
        very large databases," Journal of Systems and Software February 2016 https://doi.org/10.1016/j.jss.2015.10.035

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
        save(oFile)
            Complete set of periodic-frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of periodic-frequent patterns will be loaded in to a dataframe
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
        python3 PFPGrowthPlus.py <inputFile> <outputFile> <minSup> <maxPer>

        Examples:
        ------
        python3 PFPGrowthPlus.py sampleTDB.txt patterns.txt 0.3 0.4   (minSup will be considered in percentage of database transactions)

        python3 PFPGrowthPlus.py sampleTDB.txt patterns.txt 3 4     (minSup will be considered in support count or frequency)

    Sample run of the imported code:
    --------------

            from PAMI.periodicFrequentPattern.basic import PFPGorwthPlus as alg

            obj = alg.PFPGrowthPlus("../basic/sampleTDB.txt", "2", "6")

            obj.startMine()

            periodicFrequentPatterns = obj.getPatterns()

            print("Total number of Periodic Frequent Patterns:", len(periodicFrequentPatterns))

            obj.save("patterns")

            Df = obj.getPatternsAsDataFrame()

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

    _minSup = str()
    _maxPer = str()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _rank = {}
    _rankedUp = {}
    _lno = 0

    def _creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable


        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            data, ts = [], []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'TS' in i:
                ts = self._iFile['TS'].tolist()
            if 'Transactions' in i:
                data = self._iFile['Transactions'].tolist()
            for i in range(len(data)):
                tr = [ts[i][0]]
                tr = tr + data[i]
                self._Database.append(tr)
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

    def _periodicFrequentOneItem(self):
        """
            calculates the support of each item in the dataset and assign the ranks to the items
            by decreasing support and returns the frequent items list

            """
        data = {}
        for tr in self._Database:
            n = int(tr[0])
            for i in range(1, len(tr)):
                if n <= self._maxPer:
                    if tr[i] not in data:
                        data[tr[i]] = [int(tr[0]), int(tr[0]), 1]
                    else:
                        data[tr[i]][0] = max(data[tr[i]][0], (int(tr[0]) - data[tr[i]][1]))
                        data[tr[i]][1] = int(tr[0])
                        data[tr[i]][2] += 1
                else:
                    if tr[i] in data:
                        lp = abs(n - data[tr[i]][1])
                        if lp > self._maxPer:
                            del data[tr[i]]
                        else:
                            data[tr[i]][0] = max(data[tr[i]][0], lp)
                            data[tr[i]][1] = int(tr[0])
                            data[tr[i]][2] += 1
        for key in data:
            data[key][0] = max(data[key][0], _lno - data[key][1])
        data = {k: [v[2], v[0]] for k, v in data.items() if v[0] <= self._maxPer and v[2] >= self._minSup}
        genList = [k for k, v in sorted(data.items(), key=lambda x: (x[1][0], x[0]), reverse=True)]
        self._rank = dict([(index, item) for (item, index) in enumerate(genList)])
        # genList=[k for k,v in sorted(data.items(),key=lambda x: (x[1][0],x[0]),reverse=True)]
        return data, genList

    def _updateTransactions(self, dict1):
        """remove the items which are not frequent from transactions and updates the transactions with rank of items

            :param dict1 : frequent items with support
            :type dict1 : dictionary
            """
        list1 = []
        for tr in self._Database:
            list2 = [int(tr[0])]
            for i in range(1, len(tr)):
                if tr[i] in dict1:
                    list2.append(self._rank[tr[i]])
            if len(list2) >= 2:
                basket = list2[1:]
                basket.sort()
                list2[1:] = basket[0:]
                list1.append(list2)
        return list1

    @staticmethod
    def _buildTree(data, info):
        """it takes the transactions and support of each item and construct the main tree with setting root
                    node as null

                        :param data : it represents the one transactions in database
                        :type data : list
                        :param info : it represents the support of each item
                        :type info : dictionary
                        """
        rootNode = _Tree()
        rootNode.info = info.copy()
        for i in range(len(data)):
            set1 = [data[i][0]]
            rootNode.addTransaction(data[i][1:], set1)
        return rootNode

    def _savePeriodic(self, itemSet):
        """
        To convert item ranks into original item names
        :param itemSet: periodic-frequent pattern
        :return: original itemSet
        """
        t1 = str()
        for i in itemSet:
            t1 = t1 + self._rankedUp[i] + " "
        return t1

    def _convert(self, value):
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

    def startMine(self):
        """
            Main method where the patterns are mined by constructing tree.

        """
        global _minSup, _maxPer, _lno
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._minSup is None:
            raise Exception("Please enter the Minimum Support")
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        self._maxPer = self._convert(self._maxPer)
        _minSup, _maxPer, _lno = self._minSup, self._maxPer, len(self._Database)
        generatedItems, pfList = self._periodicFrequentOneItem()
        updatedTransactions = self._updateTransactions(generatedItems)
        for x, y in self._rank.items():
            self._rankedUp[y] = x
        info = {self._rank[k]: v for k, v in generatedItems.items()}
        Tree = self._buildTree(updatedTransactions, info)
        patterns = Tree.generatePatterns([])
        self._finalPatterns = {}
        for i in patterns:
            x = self._savePeriodic(i[0])
            self._finalPatterns[x] = i[1]
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryRSS = float()
        self._memoryUSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("periodic-frequent patterns were generated successfully using PFPGrowth++ algorithm ")

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
        """Storing final periodic-frequent patterns in a dataframe

        :return: returning periodic-frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataframe = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b[0], b[1]])
            dataframe = _ab._pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
        return dataframe

    def save(self, outFile):
        """Complete set of periodic-frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.replace(' ','\t').strip() + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of periodic-frequent patterns after completion of the mining process

        :return: returning periodic-frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        print("Total number of Periodic Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = PFPGrowthPlus(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = PFPGrowthPlus(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        print("Total number of Periodic-Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
