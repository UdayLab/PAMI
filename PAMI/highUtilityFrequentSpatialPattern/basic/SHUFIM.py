from PAMI.highUtilityFrequentSpatialPattern.basic import abstract as _ab
from functools import cmp_to_key as _comToKey

class _Transaction:
    """
        A class to store Transaction of a database

    Attributes:
    ----------
        items: list
            A list of items in transaction 
        utilities: list
            A list of utilites of items in transaction
        transactionUtility: int
            represent total sum of all utilities in the database
        pmus: list
            represent the pmu (probable maximum utility) of each element in the transaction
        prefixutility:
            prefix Utility values of item
        offset:
            an offset pointer, used by projected transactions
        support:
            maintains the support of the transaction
    Methods:
    --------
        projectedTransaction(offsetE):
            A method to create new Transaction from existing till offsetE
        getItems():
            return items in transaction
        getUtilities():
            return utilities in transaction
        getPmus():
            return pmus in transaction
        getLastPosition():
            return last position in a transaction
        removeUnpromisingItems():
            A method to remove items with low Utility than minUtil
        insertionSort():
            A method to sort all items in the transaction
        getSupport():
            returns the support of the transaction
    """
    offset = 0
    prefixUtility = 0
    support = 1
    
    def __init__(self, items, utilities, transactionUtility, pmus=None):
        self.items = items
        self.utilities = utilities
        self.transactionUtility = transactionUtility
        if pmus is not None:
            self.pmus = pmus
        self.support = 1

    def projectTransaction(self, offsetE):
        """
            A method to create new Transaction from existing till offsetE

        Parameters:
        ----------
            :param offsetE: an offset over the original transaction for projecting the transaction
            :type offsetE: int
        """
        newTransaction = _Transaction(self.items, self.utilities, self.transactionUtility)
        utilityE = self.utilities[offsetE]
        newTransaction.prefixUtility = self.prefixUtility + utilityE
        newTransaction.transactionUtility = self.transactionUtility - utilityE
        newTransaction.support = self.support
        for i in range(self.offset, offsetE):
            newTransaction.transactionUtility -= self.utilities[i]
        newTransaction.offset = offsetE + 1
        return newTransaction

    def getItems(self):
        """
            A method to return items in transaction
        """
        return self.items

    def getPmus(self):
        """
            A method to return pmus in transaction
        """
        return self.pmus

    def getUtilities(self):
        """
            A method to return utilities in transaction
        """
        return self.utilities

    # get the last position in this transaction
    def getLastPosition(self):
        """
            A method to return last position in a transaction
        """
        return len(self.items) - 1

    def getSupport(self):
        """
            A method to return support of a transaction (number of transactions in the original database having the items present in this transactions)
        """
        return self.support

    def removeUnpromisingItems(self, oldNamesToNewNames):
        """
            A method to remove items with low Utility than minUtil
            :param oldNamesToNewNames: A map represent old namses to new names
            :type oldNamesToNewNames: map
        """
        tempItems = []
        tempUtilities = []
        for idx, item in enumerate(self.items):
            if item in oldNamesToNewNames:
                tempItems.append(oldNamesToNewNames[item])
                tempUtilities.append(self.utilities[idx])
            else:
                self.transactionUtility -= self.utilities[idx]
        self.items = tempItems
        self.utilities = tempUtilities
        self.insertionSort()

    def insertionSort(self):
        """
            A method to sort items in order
        """
        for i in range(1, len(self.items)):
            key = self.items[i]
            utilityJ = self.utilities[i]
            j = i - 1
            while j >= 0 and key < self.items[j]:
                self.items[j + 1] = self.items[j]
                self.utilities[j + 1] = self.utilities[j]
                j -= 1
            self.items[j + 1] = key
            self.utilities[j + 1] = utilityJ


class _Dataset:
    """
        A class represent the list of transactions in this dataset

    Attributes:
    ----------
        transactions :
            the list of transactions in this dataset
        maxItem:
            the largest item name
        
    methods:
    --------
        createTransaction(line):
            Create a transaction object from a line from the input file
        getMaxItem():
            return Maximum Item
        getTransactions():
            return transactions in database

    """
    transactions = []
    maxItem = 0
    
    def __init__(self, datasetpath, sep):
        self.strToint = {}
        self.intTostr = {}
        self.cnt = 1
        self.sep = sep
        with open(datasetpath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.transactions.append(self.createTransaction(line))
        f.close()

    def createTransaction(self, line):
        """
            A method to create Transaction from dataset given
            
            Attributes:
            -----------
            :param line: represent a single line of database
            :param line: represent a single line of database
            :type line: string
            :return : Transaction
            :rtype: Transaction
        """
        #print(line)
        transList = line.strip().split(':')
        transactionUtility = int(transList[1])
        itemsString = transList[0].strip().split(self.sep)
        utilityString = transList[2].strip().split(self.sep)
        pmuString = transList[3].strip().split(self.sep)
        items = []
        utilities = []
        pmus = []
        for idx, item in enumerate(itemsString):
            if (self.strToint).get(item) is None:
                self.strToint[item] = self.cnt
                self.intTostr[self.cnt] = item
                self.cnt += 1
            itemInt = self.strToint.get(item)
            if itemInt > self.maxItem:
                self.maxItem = itemInt
            items.append(itemInt)
            utilities.append(int(utilityString[idx]))
            pmus.append(int(pmuString[idx]))
        return _Transaction(items, utilities, transactionUtility, pmus)

    def getMaxItem(self):
        """
            A method to return name of largest item
        """
        return self.maxItem

    def getTransactions(self):
        """
            A method to return transactions from database
        """
        return self.transactions


class SHUFIM(_ab._utilityPatterns):
    """
      Spatial High Utility Frequent ItemSet Mining (SHUFIM) aims to discover all itemSets in a spatioTemporal database
       that satisfy the user-specified minimum utility, minimum support and maximum distance constraints
    Reference:
    ---------
        10.1007/978-3-030-37188-3_17

    Attributes:
    -----------
        iFile : file
            Name of the input file to mine complete set of frequent patterns
        nFile : file
            Name of the Neighbours file that contain neighbours of items
        oFile : file
            Name of the output file to store complete set of frequent patterns
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        minUtil : int
            The user given minUtil
        minSup : float
            The user given minSup value
        highUtilityFrequentSpatialItemSets: map
            set of high utility itemSets
        candidateCount: int
             Number of candidates 
        utilityBinArrayLU: list
             A map to hold the pmu values of the items in database
        utilityBinArraySU: list
            A map to hold the subtree utility values of the items is database
        oldNamesToNewNames: list
            A map to hold the subtree utility values of the items is database
        newNamesToOldNames: list
            A map to store the old name corresponding to new name
        Neighbours : map
            A dictionary to store the neighbours of a item
        maxMemory: float
            Maximum memory used by this program for running
        patternCount: int
            Number of SHUFI's (Spatial High Utility Frequent Itemsets)
        itemsToKeep: list
            keep only the promising items ie items whose supersets can be required patterns
        itemsToExplore: list
            keep items that subtreeUtility grater than minUtil

    Methods :
    -------
        startMine()
                Mining process will start from here
        getPatterns()
                Complete set of patterns will be retrieved with this function
        savePatterns(oFile)
                Complete set of frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
                Complete set of frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
                Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
                Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
               Total amount of runtime taken by the mining process will be retrieved from this function
        calculateNeighbourIntersection(self, prefixLength)
               A method to return common Neighbours of items
        backtrackingEFIM(transactionsOfP, itemsToKeep, itemsToExplore, prefixLength)
               A method to mine the SHUIs Recursively
        useUtilityBinArraysToCalculateUpperBounds(transactionsPe, j, itemsToKeep, neighbourhoodList)
               A method to  calculate the sub-tree utility and local utility of all items that can extend itemSet P and e
        output(tempPosition, utility)
               A method ave a high-utility itemSet to file or memory depending on what the user chose
        isEqual(transaction1, transaction2)
               A method to Check if two transaction are identical
        intersection(lst1, lst2)
               A method that return the intersection of 2 list
        useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(dataset)
              Scan the initial database to calculate the subtree utility of each items using a utility-bin array
        sortDatabase(self, transactions)
              A Method to sort transaction in the order of PMU
        sortTransaction(self, trans1, trans2)
              A Method to sort transaction in the order of PMU
        useUtilityBinArrayToCalculateLocalUtilityFirstTime(self, dataset)
             A method to scan the database using utility bin array to calculate the pmus

    Executing the code on terminal :
    -------
        Format: python3 SHUFIM.py <inputFile> <outputFile> <Neighbours> <minUtil> <minSup> <sep>
        Examples: python3 SHUFIM.py sampleTDB.txt output.txt sampleN.txt 35 20 (it will consider "\t" as separator)
                  python3 SHUFIM.py sampleTDB.txt output.txt sampleN.txt 35 20 , (it will consider "," as separator)

    Sample run of importing the code:
    -------------------------------
        
        from PAMI.highUtilityFrequentSpatialPattern.basic import SHUFIM as alg

        obj=alg.SHUFIM("input.txt","Neighbours.txt",35,20)

        obj.startMine()

        patterns = obj.getPatterns()

        print("Total number of Spatial high utility frequent Patterns:", len(patterns))

        obj.savePatterns("output")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    Credits:
    -------
            The complete program was written by Pradeep Pallikila under the supervision of Professor Rage Uday Kiran.
    """
    _candidateCount = 0
    _utilityBinArrayLU = {}
    _utilityBinArraySU = {}
    _oldNamesToNewNames = {}
    _newNamesToOldNames = {}
    _singleItemSetsSupport = {}
    _singleItemSetsUtility = {}
    _strToint = {}
    _intTostr = {}
    _Neighbours = {}
    _temp = [0] * 5000
    _maxMemory = 0
    _startTime = float()
    _endTime = float()
    _minSup = str()
    _maxPer = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _nFile = " "
    _sep = "\t"
    _minUtil = 0
    _memoryUSS = float()
    _memoryRSS = float()
    
    def __init__(self, iFile, nFile, minUtil, minSup, sep="\t"):
        super().__init__(iFile, nFile, minUtil, minSup, sep)

    def startMine(self):
        self._startTime = _ab._time.time()
        self._patternCount = 0
        self._finalPatterns = {}
        self._dataset = _Dataset(self._iFile, self._sep)
        self._singleItemSetsSupport = _ab._defaultdict(int)
        self._singleItemSetsUtility = _ab._defaultdict(int)
        self._minUtil = int(self._minUtil)
        self._minSup = int((self._minSup * len(self._dataset.getTransactions())) / 100)
        #print("######################################")
        #print("given minimum support is", self.minSup)
        #print("given minimum utility is", self.minUtil)
        with open(self._nFile, 'r') as o:
            lines = o.readlines()
            for line in lines:
                line = line.split("\n")[0]
                line_split = line.split(self._sep)
                item = self._dataset.strToint.get(line_split[0])
                lst = []
                for i in range(1, len(line_split)):
                    lst.append(self._dataset.strToint.get(line_split[i]))
                self._Neighbours[item] = lst
        o.close()
        InitialMemory = _ab._psutil.virtual_memory()[3]
        self._useUtilityBinArrayToCalculateLocalUtilityFirstTime(self._dataset)
        _itemsToKeep = []
        for key in self._utilityBinArrayLU.keys():
            if self._utilityBinArrayLU[key] >= self._minUtil and self._singleItemSetsSupport[key] >= self._minSup:
                _itemsToKeep.append(key)
        # sorting items in decreasing order of their utilities
        _itemsToKeep = sorted(_itemsToKeep, key=lambda x: self._singleItemSetsUtility[x], reverse=True)
        _currentName = 1
        for idx, item in enumerate(_itemsToKeep):
            self._oldNamesToNewNames[item] = _currentName
            self._newNamesToOldNames[_currentName] = item
            _itemsToKeep[idx] = _currentName
            _currentName += 1
        for transaction in self._dataset.getTransactions():
            transaction.removeUnpromisingItems(self._oldNamesToNewNames)
        self._sortDatabase(self._dataset.getTransactions())
        _emptyTransactionCount = 0
        for transaction in self._dataset.getTransactions():
            if len(transaction.getItems()) == 0:
                _emptyTransactionCount += 1
        self._dataset.transactions = self._dataset.transactions[_emptyTransactionCount:]
        # calculating neighborhood suffix utility values
        _secondary = []
        for idx, item in enumerate(_itemsToKeep):
            _cumulativeUtility = self._singleItemSetsUtility[self._newNamesToOldNames[item]]
            if self._newNamesToOldNames[item] in self._Neighbours:
                neighbors = [self._oldNamesToNewNames[y] for y in self._Neighbours[self._newNamesToOldNames[item]] if y in self._oldNamesToNewNames]
                for i in range(idx+1, len(_itemsToKeep)):
                    _nextItem = _itemsToKeep[i]
                    if _nextItem in neighbors:
                        _cumulativeUtility += self._singleItemSetsUtility[self._newNamesToOldNames[_nextItem]]
            if _cumulativeUtility >= self._minUtil:
                _secondary.append(item)         
        self._useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(self._dataset)
        _itemsToExplore = []
        for item in _secondary:
            if self._utilityBinArraySU[item] >= self._minUtil:
                _itemsToExplore.append(item)
        _commonitems = []
        for i in range(self._dataset.maxItem):
            _commonitems.append(i)
        self._backtrackingEFIM(self._dataset.getTransactions(), _itemsToKeep, _itemsToExplore, 0)
        _finalMemory = _ab._psutil.virtual_memory()[3]
        memory = (_finalMemory - InitialMemory) / 10000
        if memory > self._maxMemory:
            self._maxMemory = memory
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print('Spatial High Utility Frequent Itemsets generated successfully using SHUFIM algorithm')

    def _backtrackingEFIM(self, transactionsOfP, itemsToKeep, itemsToExplore, prefixLength):
        """
            A method to mine the SHUFIs Recursively

            Attributes:
            ----------
            :param transactionsOfP: the list of transactions containing the current prefix P
            :type transactionsOfP: list 
            :param itemsToKeep: the list of secondary items in the p-projected database
            :type itemsToKeep: list
            :param itemsToExplore: the list of primary items in the p-projected database
            :type itemsToExplore: list
            :param prefixLength: current prefixLength
            :type prefixLength: int
        """
        self._candidateCount += len(itemsToExplore)
        for idx, e in enumerate(itemsToExplore):
            initialMemory = _ab._psutil.virtual_memory()[3]
            transactionsPe = []
            utilityPe = 0
            supportPe = 0
            previousTransaction = []
            consecutiveMergeCount = 0
            for transaction in transactionsOfP:
                items = transaction.getItems()
                if e in items:
                    positionE = items.index(e)
                    if transaction.getLastPosition() == positionE:
                        utilityPe += transaction.getUtilities()[positionE] + transaction.prefixUtility
                        supportPe += transaction.getSupport()
                    else:
                        projectedTransaction = transaction.projectTransaction(positionE)
                        utilityPe += projectedTransaction.prefixUtility
                        if previousTransaction == []:
                            previousTransaction = projectedTransaction
                        elif self._isEqual(projectedTransaction, previousTransaction):
                            if consecutiveMergeCount == 0:
                                items = previousTransaction.items[previousTransaction.offset:]
                                utilities = previousTransaction.utilities[previousTransaction.offset:]
                                support = previousTransaction.getSupport()
                                itemsCount = len(items)
                                positionPrevious = 0
                                positionProjection = projectedTransaction.offset
                                while positionPrevious < itemsCount:
                                    utilities[positionPrevious] += projectedTransaction.utilities[positionProjection]
                                    positionPrevious += 1
                                    positionProjection += 1
                                previousTransaction.prefixUtility += projectedTransaction.prefixUtility
                                sumUtilities = previousTransaction.prefixUtility
                                previousTransaction = _Transaction(items, utilities, previousTransaction.transactionUtility + projectedTransaction.transactionUtility)
                                previousTransaction.prefixUtility = sumUtilities
                                previousTransaction.support = support
                                previousTransaction.support += projectedTransaction.getSupport()
                            else:
                                positionPrevious = 0
                                positionProjected = projectedTransaction.offset
                                itemsCount = len(previousTransaction.items)
                                while positionPrevious < itemsCount:
                                    previousTransaction.utilities[positionPrevious] += projectedTransaction.utilities[
                                        positionProjected]
                                    positionPrevious += 1
                                    positionProjected += 1
                                previousTransaction.transactionUtility += projectedTransaction.transactionUtility
                                previousTransaction.prefixUtility += projectedTransaction.prefixUtility
                                previousTransaction.support += projectedTransaction.getSupport()
                            consecutiveMergeCount += 1
                        else:
                            transactionsPe.append(previousTransaction)
                            supportPe += previousTransaction.getSupport()
                            previousTransaction = projectedTransaction
                            consecutiveMergeCount = 0
                    transaction.offset = positionE
            if previousTransaction != []:
                transactionsPe.append(previousTransaction)
                supportPe += previousTransaction.getSupport()
            self._temp[prefixLength] = self._newNamesToOldNames[e]
            if utilityPe >= self._minUtil and supportPe >= self._minSup:
                self._output(prefixLength, utilityPe, supportPe)
            if supportPe >= self._minSup:
                neighbourhoodList = self._calculateNeighbourIntersection(prefixLength)
                #print(neighbourhoodList)
                self._useUtilityBinArraysToCalculateUpperBounds(transactionsPe, idx, itemsToKeep, neighbourhoodList)
                newItemsToKeep = []
                newItemsToExplore = []
                for l in range(idx + 1, len(itemsToKeep)):
                    itemK = itemsToKeep[l]
                    if self._utilityBinArraySU[itemK] >= self._minUtil:
                        if itemK in neighbourhoodList:
                            newItemsToExplore.append(itemK)
                            newItemsToKeep.append(itemK)
                    elif self._utilityBinArrayLU[itemK] >= self._minUtil:
                        if itemK in neighbourhoodList:
                            newItemsToKeep.append(itemK)
                self._backtrackingEFIM(transactionsPe, newItemsToKeep, newItemsToExplore, prefixLength + 1)
            finalMemory = _ab._psutil.virtual_memory()[3]
            memory = (finalMemory - initialMemory) / 10000
            if self.maxMemory < memory:
                self.maxMemory = memory

    def _useUtilityBinArraysToCalculateUpperBounds(self, transactionsPe, j, itemsToKeep, neighbourhoodList):
        """
            A method to  calculate the sub-tree utility and local utility of all items that can extend itemSet P U {e}

            Attributes:
            -----------
            :param transactionsPe: transactions the projected database for P U {e}
            :type transactionsPe: list
            :param j:he position of j in the list of promising items
            :type j:int
            :param itemsToKeep :the list of promising items
            :type itemsToKeep: list

        """
        for i in range(j + 1, len(itemsToKeep)):
            item = itemsToKeep[i]
            self._utilityBinArrayLU[item] = 0
            self._utilityBinArraySU[item] = 0
        for transaction in transactionsPe:
            length = len(transaction.getItems())
            i = length - 1
            while i >= transaction.offset:
                item = transaction.getItems()[i]
                if item in itemsToKeep:
                    remainingUtility = 0
                    if self._newNamesToOldNames[item] in self._Neighbours:
                        itemNeighbours = self._Neighbours[self._newNamesToOldNames[item]]
                        for k in range(i, length):
                            transaction_item = transaction.getItems()[k]
                            if self._newNamesToOldNames[transaction_item] in itemNeighbours and transaction_item in neighbourhoodList:
                                remainingUtility += transaction.getUtilities()[k]

                    remainingUtility += transaction.getUtilities()[i]
                    self._utilityBinArraySU[item] += remainingUtility + transaction.prefixUtility
                    self._utilityBinArrayLU[item] += transaction.transactionUtility + transaction.prefixUtility
                i -= 1

    def _calculateNeighbourIntersection(self, prefixLength):
        """
            A method to find common Neighbours
            Attributes:
            ----------
                :param prefixLength: the prefix itemSet
                :type prefixLength:int

        """
        intersectionList = self._Neighbours.get(self._temp[0])
        for i in range(1, prefixLength+1):
            intersectionList = self._intersection(self._Neighbours[self._temp[i]], intersectionList)
        finalIntersectionList = []
        if intersectionList is None:
            return finalIntersectionList
        for item in intersectionList:
            if item in self._oldNamesToNewNames:
                finalIntersectionList.append(self._oldNamesToNewNames[item])
        return finalIntersectionList
    
    def _output(self, tempPosition, utility, support):
        """
         A method save all high-utility itemSet to file or memory depending on what the user chose

         Attributes:
         ----------
         :param tempPosition: position of last item
         :type tempPosition : int 
         :param utility: total utility of itemSet
         :type utility: int
         :param support: support of an itemSet
         :type support: int
        """
        self._patternCount += 1
        s1 = ""
        for i in range(0, tempPosition+1):
            s1 += self._dataset.intTostr.get((self._temp[i]))
            if i != tempPosition:
                s1 += " "
        self._finalPatterns[s1] = str(utility) + ":" + str(support)

    def _isEqual(self, transaction1, transaction2):
        """
         A method to Check if two transaction are identical

         Attributes:
         ----------
         :param  transaction1: the first transaction
         :type  transaction1: Transaction
         :param  transaction2:   the second transaction
         :type  transaction2: Transaction
         :return : whether both are identical or not
         :rtype: bool
        """

        length1 = len(transaction1.items) - transaction1.offset
        length2 = len(transaction2.items) - transaction2.offset
        if length1 != length2:
            return False
        position1 = transaction1.offset
        position2 = transaction2.offset
        while position1 < len(transaction1.items):
            if transaction1.items[position1] != transaction2.items[position2]:
                return False
            position1 += 1
            position2 += 1
        return True
    
    def _intersection(self, lst1, lst2):
        """
            A method that return the intersection of 2 list
            :param  lst1: items neighbour to item1
            :type lst1: list
            :param lst2: items neighbour to item2
            :type lst2: list
            :return :intersection of two lists
            :rtype : list
        """
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    def _useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(self, dataset):
        """
        Scan the initial database to calculate the subtree utility of each items using a utility-bin array

        Attributes:
        ----------
        :param dataset: the transaction database
        :type dataset: Dataset
        """
        for transaction in dataset.getTransactions():
            items = transaction.getItems()
            utilities = transaction.getUtilities()
            for idx, item in enumerate(items):
                if item not in self._utilityBinArraySU:
                    self._utilityBinArraySU[item] = 0
                if self._newNamesToOldNames[item] not in self._Neighbours:
                    self._utilityBinArraySU[item] += utilities[idx]
                    continue
                i = idx + 1
                sumSu = utilities[idx]
                while i < len(items):
                    if self._newNamesToOldNames[items[i]] in self._Neighbours[self._newNamesToOldNames[item]]:
                        sumSu += utilities[i]
                    i += 1
                self._utilityBinArraySU[item] += sumSu

    def _sortDatabase(self, transactions):
        """
            A Method to sort transaction in the order of PMU

            Attributes:
            ----------
            :param transactions: transaction of items
            :type transactions: Transaction 
            :return: sorted transaction
            :rtype: Transaction
        """
        cmp_items = _comToKey(self._sortTransaction)
        transactions.sort(key=cmp_items)

    def _sortTransaction(self, trans1, trans2):
        """
            A Method to sort transaction in the order of PMU

            Attributes:
            ----------
            :param trans1: the first transaction 
            :type trans1: Transaction 
            :param trans2:the second transaction 
            :type trans2: Transaction
            :return: sorted transaction
            :rtype:    Transaction
        """
        trans1_items = trans1.getItems()
        trans2_items = trans2.getItems()
        pos1 = len(trans1_items) - 1
        pos2 = len(trans2_items) - 1
        if len(trans1_items) < len(trans2_items):
            while pos1 >= 0:
                sub = trans2_items[pos2] - trans1_items[pos1]
                if sub != 0:
                    return sub
                pos1 -= 1
                pos2 -= 1
            return -1
        elif len(trans1_items) > len(trans2_items):
            while pos2 >= 0:
                sub = trans2_items[pos2] - trans1_items[pos1]
                if sub != 0:
                    return sub
                pos1 -= 1
                pos2 -= 1
            return 1
        else:
            while pos2 >= 0:
                sub = trans2_items[pos2] - trans1_items[pos1]
                if sub != 0:
                    return sub
                pos1 -= 1
                pos2 -= 1
            return 0

    def _useUtilityBinArrayToCalculateLocalUtilityFirstTime(self, dataset):
        """
            A method to scan the database using utility bin array to calculate the pmus
            Attributes:
            ----------
            :param dataset: the transaction database
            :type dataset: database

        """
        for transaction in dataset.getTransactions():
            for idx, item in enumerate(transaction.getItems()):
                self._singleItemSetsSupport[item] += 1
                self._singleItemSetsUtility[item] += transaction.getUtilities()[idx]
                if item in self._utilityBinArrayLU:
                    self._utilityBinArrayLU[item] += transaction.getPmus()[idx]
                else:
                    self._utilityBinArrayLU[item] = transaction.getPmus()[idx]

    def getPatternsAsDataFrame(self):
        """Storing final patterns in a dataframe

        :return: returning patterns in a dataframe
        :rtype: pd.DataFrame
        """
        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Utility:Support'])

        return dataFrame
    
    def getPatterns(self):
        """ Function to send the set of patterns after completion of the mining process

        :return: returning patterns
        :rtype: dict
        """
        return self._finalPatterns

    def savePatterns(self, outFile):
        """Complete set of patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = str(x) + " : " + str(y)
            writer.write("%s \n" % patternsAndSupport)

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
        return self._endTime-self._startTime


if __name__ == '__main__':
    _ap = str()
    if len(_ab._sys.argv) == 6 or len(_ab._sys.argv) == 7:
        if len(_ab._sys.argv) == 7:
            _ap = SHUFIM(_ab._sys.argv[1], _ab._sys.argv[3], int(_ab._sys.argv[4]), float(_ab._sys.argv[5]), _ab._sys.argv[6])
        if len(_ab._sys.argv) == 6:
            _ap = SHUFIM(_ab._sys.argv[1], _ab._sys.argv[3], int(_ab._sys.argv[4]), float(_ab._sys.argv[5]))
        _ap.startMine()
        _patterns = _ap.getPatterns()
        print("Total number of Spatial High Utility Frequent Patterns:", len(_patterns))
        _ap.savePatterns(_ab._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in seconds:", _run)
        #print("######################################")
    else:
        _ap = SHUFIM('/Users/likhitha/Downloads/HUIS/main_1.txt',
                      '/Users/likhitha/Downloads/HUIS/bms_neighbourhoodFile_1.txt',
                      10, 10, ' ')
        _ap.startMine()
        _patterns = _ap.getPatterns()
        print("Total number of Spatial High Utility Patterns:", len(_patterns))
        _ap.savePatterns('/Users/likhitha/Downloads/HUIS/output.txt')
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in seconds:", _run)
        print("Error! The number of input parameters do not match the total number of parameters provided")
