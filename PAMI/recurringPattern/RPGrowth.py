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
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.PAMI.periodicFrequentPattern.recurring.

from PAMI.recurringPattern import abstract as _ab


_maxPer = float()
_minPS = float()
_minRec = float()
_lno = int()


class _Node(object):
    """
        A class used to represent the node of frequentPatternTree

        Attributes:
        ----------
            item : int or None
                Storing item of a node
            timeStamps : list
                To maintain the timestamps of a database at the end of the branch
            parent : node
                To maintain the parent of every node
            children : list
                To maintain the children of a node

        Methods:
        -------
            addChild(itemName)
                Storing the children to their respective parent nodes
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
        self.parent = None
        self.timeStamps = []

    def addChild(self, node):
        """ To add the children to a node

            :param node: parent node in the tree
        """

        self.children[node.item] = node
        node.parent = self


class _Tree(object):
    """
        A class used to represent the frequentPatternGrowth tree structure

        Attributes:
        ----------
            root : Node
                Represents the root node of the tree
            summaries : dictionary
                Storing the nodes with same item name
            info : dictionary
                Stores the support of the items


        Methods:
        -------
            addTransactions(Database)
                Creating transaction as a branch in Recurring PatternTree
            getConditionalPatterns(Node)
                Generates the conditional patterns from tree for specific node
            conditionalTransaction(prefixPaths,Support)
                Takes the prefixPath of a node and support at child of the path and extract the frequent patterns from
                prefixPaths and generates prefixPaths with items which are frequent
            remove(Node)
                Removes the node from tree once after generating all the patterns respective to the node
            generatePatterns(Node)
                Starts from the root node of the tree and mines the periodic-frequent patterns

        """

    def __init__(self):
        self.root = _Node(None, {})
        self.summaries = {}
        self.info = {}

    def addTransaction(self, transaction, tid):
        """     Adding a transaction into tree

                :param transaction: To represent the complete database
                :type transaction: list
                :param tid: To represent the timestamp of a database
                :type tid: list
                :return: rp-tree
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
        """Generates all the conditional patterns of a respective node

            :param alpha: To represent a Node in the tree
            :type alpha: Node
            :return: A tuple consisting of finalPatterns, conditional pattern base and information
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
        finalPatterns, finalSets, info = self.conditionalDatabases(finalPatterns, finalSets)
        return finalPatterns, finalSets, info

    @staticmethod
    def generateTimeStamps(node):
        """To get the timestamps of a node

        :param node: A node in the tree
        :return: Timestamps of a node
        """

        finalTimeStamps = node.timeStamps
        return finalTimeStamps

    def removeNode(self, nodeValue):
        """ Removing the node from tree

            :param nodeValue: To represent a node in the tree
            :type nodeValue: node
            :return: Tree with their nodes updated with timestamps
        """

        for i in self.summaries[nodeValue]:
            i.parent.timeStamps = i.parent.timeStamps + i.timeStamps
            del i.parent.children[nodeValue]

    def getTimeStamps(self, alpha):
        """ To get all the timestamps of the nodes which share same item name

            :param alpha: Node in a tree
            :return: Timestamps of a  node
        """
        temporary = []
        for i in self.summaries[alpha]:
            temporary += i.timeStamps
        return temporary

    @staticmethod
    def getSupportAndPeriod(timeStamps):
        """To calculate the recurrence and support

        :param timeStamps: Timestamps of an item set
        :return: recurring intervals with corresponding periodic support, summation of support of periodic intervals, support
        """

        global maxPer,minPS
        timeStamps.sort()
        cur = ' '
        st = ' '
        end = ' '
        if len(timeStamps) > 0:
            cur = timeStamps[0]
            st = timeStamps[0]
            end = timeStamps[0]
        ps = 0
        lps = 1
        recli = []
        for i in range(1, len(timeStamps)):
            if abs(timeStamps[i] - cur) <= maxPer:
                lps += 1
            else:
                if lps >= minPS:
                    recli.append([st, end, lps])
                    ps += lps
                lps = 1
                st = timeStamps[i]
            cur = timeStamps[i]
            end = cur
        if lps >= minPS:
            recli.append([st, end, lps])
            ps+=lps
        # print(recli)
        return [recli, ps, len(timeStamps)]

    def conditionalDatabases(self, conditionalPatterns, conditionalTimeStamps):
        """ It generates the conditional patterns with periodic-frequent items

            :param conditionalPatterns: conditionalPatterns generated from conditionPattern method of a respective node
            :type conditionalPatterns: list
            :param conditionalTimeStamps: Represents the timestamps of a conditional patterns of a node
            :type conditionalTimeStamps: list
            :returns: Returns conditional transactions by removing non recurring items
        """

        global maxPer, minPS, minRec
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
        # print(updatedDictionary)
        updatedDictionary = {k: [v[0],v[2]] for k, v in updatedDictionary.items() if v[1] >= (minPS*minRec)}
        count = 0
        for p in conditionalPatterns:
            p1 = [v for v in p if v in updatedDictionary]
            trans = sorted(p1, key=lambda x: (updatedDictionary.get(x)[1], -x), reverse=True)
            if len(trans) > 0:
                pat.append(trans)
                timeStamps.append(conditionalTimeStamps[count])
            count += 1
        return pat, timeStamps, updatedDictionary

    def generatePatterns(self, prefix):
        """ Generates the patterns

            :param prefix: Forms the combination of items
            :type prefix: list
            :returns: yields patterns with their recurrence and support
        """

        for i in sorted(self.summaries, key=lambda x: (self.info.get(x)[1], -x)):
            pattern = prefix[:]
            pattern.append(i)
            if len(self.info.get(i)[0])>=minRec:
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


class RPGrowth(_ab._recurringPatterns):
    """ RPGrowth is one of the fundamental algorithm to discover recurring patterns in a transactional database.

   

    Attributes:
    ----------
        iFile : file
            Name of the Input file or path of the input file
        oFile : file
            Name of the output file or path of the output file
        maxPer: int or float or str
            The user can specify maxPer either in count or proportion of database size.
            If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxPer=10 will be treated as integer, while maxPer=10.0 will be treated as float
        minPS: int or float or str
            The user can specify minPS either in count or proportion of database size.
            If the program detects the data type of minPS is integer, then it treats minPS is expressed in count.
            Otherwise, it will be treated as float.
            Example: minPS=10 will be treated as integer, while minPS=10.0 will be treated as float
        minRec: int or float or str
            The user has to specify minRec in count.
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
            To represent the total no of transaction
        tree : class
            To represents the Tree class
        itemSetCount : int
            To represents the total no of patterns
        finalPatterns : dict
            To store the complete patterns

    Methods:
    -------
        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        savePatterns(oFile)
            Complete set of periodic-frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of periodic-frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        creatingItemSets(fileName)
            Scans the dataset and stores in a list format
        OneItems()
            Extracts the possible recurring items of size one from database
        updateDatabases()
            Update the database by removing non recurring items and sort the Database by item decreased support
        buildTree()
            After updating the Database, remaining items will be added into the tree by setting root node as null
        convert()
            to convert the user specified value

    Executing the code on terminal:
    -------
        Format:
        ------
        python3 RPGrowth.py <inputFile> <outputFile> <maxPer> <minPS> <minRec>

        Examples:
        --------
        python3 RPGrowth.py sampleTDB.txt patterns.txt 0.3 0.4 2   (maxPer and minPS  will be considered in percentage of database
        transactions and minRec is integer)

        python3 RPGrowth.py sampleTDB.txt patterns.txt 3 4 2  (maxPer and minPS  will be considered in support count or frequency and minRec is integer)

    Sample run of importing the code:
    -------------------

            from PAMI.periodicFrequentPattern.recurring import RPGrowth as alg

            obj = alg.RPGrowth(iFile, maxPer, minPS, minRec)

            obj.startMine()

            recurringPatterns = obj.getPatterns()

            print("Total number of Recurring Patterns:", len(recurringPatterns))

            obj.savePatterns(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

        Credits:
        -------
            The complete program was written by C. Saideep under the supervision of Professor Rage Uday Kiran.\n

    """
    _startTime = float()
    _endTime = float()
    _minPS = str()
    _maxPer = float()
    _minRec = str()
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
        """ Storing the complete values of the database/input file in a database variable
        """

        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
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

    def _OneItems(self):
        """ Calculates the maxRec and support of each item in the database and assign ranks to the items
            by decreasing support and returns the RP-list

            :returns: return the RP-list
        """
        #global rank
        data = {}
        for tr in self._Database:
            for i in range(1, len(tr)):
                if tr[i] not in data:
                    data[tr[i]] = [[], int(tr[0]), int(tr[0]), 1, 0, 1]
                else:
                    lp = int(tr[0]) - data[tr[i]][2]
                    if lp <= self._maxPer:
                        data[tr[i]][3] += 1

                    else:
                        if data[tr[i]][3] >= self._minPS:
                            data[tr[i]][0].append([data[tr[i]][1], data[tr[i]][2], data[tr[i]][3]])
                            data[tr[i]][4] += data[tr[i]][3]
                        data[tr[i]][3] = 1
                        data[tr[i]][1] = int(tr[0])
                    data[tr[i]][2] = int(tr[0])
                    data[tr[i]][5] += 1
            # print(data)
           
        for ri in data:
            if data[ri][3] >= self._minPS:
                data[ri][0].append([data[ri][1], data[ri][2], data[ri][3]])
                data[ri][4] += data[ri][3]
        data = {k: [v[0], v[5]] for k, v in data.items() if v[4] >= (self._minPS*self._minRec)}
        genList = [k for k, v in sorted(data.items(), key=lambda x: (x[1][1], x[0]), reverse=True)]
        self._rank = dict([(index, item) for (item, index) in enumerate(genList)])
        return data, genList

    def _updateDatabases(self, dict1):
        """ Remove the items which does not  satisfy maxRec from database and updates the database with rank of items

            :param dict1: Recurring items with support and recurrence
            :type dict1: dictionary
            :return: Sorted and updated transactions
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
                # print(list2)
        return list1

    @staticmethod
    def _buildTree(data, info):
        """ It takes the database and construct the main tree by setting root node as a null

            :param data: it represents the one items in database
            :type data: list
            :param info: it represents the support and recurrence of each item
            :type info: dictionary
            :return: returns root node of tree
        """

        rootNode = _Tree()
        rootNode.info = info.copy()
        for i in range(len(data)):
            set1 = [data[i][0]]
            rootNode.addTransaction(data[i][1:], set1)
        return rootNode

    def _savePeriodic(self, itemSet):
        """ To convert the ranks of items in to their original item names

            :param itemSet: recurring pattern
            :return: recurring pattern with original item names
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
        """ Mining process will start from this function
        """

        global _minPS, _minRec, _maxPer, _lno
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        self._creatingItemSets()
        self._minPS = self._convert(self._minPS)
        self._maxPer = self._convert(self._maxPer)
        self._minRec = int(self._minRec)
        self._finalPatterns = {}
        _maxPer, _minPS, _minRec, _lno = self._maxPer, self._minPS, self._minRec, len(self._Database)
        generatedItems, pfList = self._OneItems()
        updatedDatabases = self._updateDatabases(generatedItems)
        for x, y in self._rank.items():
            self._rankedUp[y] = x
        info = {self._rank[k]: v for k, v in generatedItems.items()}
        Tree = self._buildTree(updatedDatabases, info)
        patterns = Tree.generatePatterns([])
        for i in patterns:
            sample = self._savePeriodic(i[0])
            self._finalPatterns[sample] = i[1]
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Recurring patterns were generated successfully using RPGrowth algorithm ")

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

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            z = []
            for k in b[1]:
                z.append({[k[0], k[1]], k[2]})
            data.append([a, b[1], len(b[1]), z])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support', 'Recurrance', 'intervals'])
        return dataFrame

    def savePatterns(self, outFile):
        """Complete set of periodic-frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            # print(x,y)
            str1 = '{'
            for z in y[0]:
                str1 += '{'+str([z[0], z[1]])+' : ' + str(z[2]) + '}'
            str1 += '}'
            s1 = x + ":" + str(y[1]) + ":" + str(len(y[0])) + ":" + str1
            writer.write("%s \n" % s1)
        writer.close()

    def getPatterns(self):
        """ Function to send the set of periodic-frequent patterns after completion of the mining process

        :return: returning periodic-frequent patterns
        :rtype: dict
        """
        return self._finalPatterns


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 6 or len(_ab._sys.argv) == 7:
        if len(_ab._sys.argv) == 7:
            _ap = RPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5], _ab._sys.argv[6])
        if len(_ab._sys.argv) == 6:
            _ap = RPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        _ap.startMine()
        _Patterns = _ap.getPatterns()
        print("Total number of Patterns:", len(_Patterns))
        _ap.savePatterns(_ab._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
