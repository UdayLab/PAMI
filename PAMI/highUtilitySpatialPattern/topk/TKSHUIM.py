from PAMI.highUtilitySpatialPattern.topk.abstract import *
from functools import cmp_to_key
import heapq

class Transaction:
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
    """
    offset = 0
    prefixUtility = 0
    
    def __init__(self, items, utilities, transactionUtility, pmus=None):
        self.items = items
        self.utilities = utilities
        self.transactionUtility = transactionUtility
        if pmus is not None:
            self.pmus = pmus

    def projectTransaction(self, offsetE):
        """
            A method to create new Transaction from existing till offsetE

        Parameters:
        ----------
            :param offsetE: an offset over the original transaction for projecting the transaction
            :type offsetE: int
        """
        new_transaction = Transaction(self.items, self.utilities, self.transactionUtility)
        utilityE = self.utilities[offsetE]
        new_transaction.prefixUtility = self.prefixUtility + utilityE
        new_transaction.transactionUtility = self.transactionUtility - utilityE
        for i in range(self.offset, offsetE):
            new_transaction.transactionUtility -= self.utilities[i]
        new_transaction.offset = offsetE + 1
        return new_transaction

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


class Dataset:
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
            :type line: string
            :return : Transaction
            :rtype: Transaction
        """
        trans_list = line.strip().split(':')
        transactionUtility = int(trans_list[1])
        itemsString = trans_list[0].strip().split(self.sep)
        utilityString = trans_list[2].strip().split(self.sep)
        pmuString = trans_list[3].strip().split(self.sep)
        items = []
        utilities = []
        pmus = []
        for idx, item in enumerate(itemsString):
            if (self.strToint).get(item) is None:
                self.strToint[item] = self.cnt
                self.intTostr[self.cnt] = item
                self.cnt += 1
            item_int = self.strToint.get(item)
            if item_int > self.maxItem:
                self.maxItem = item_int
            items.append(item_int)
            utilities.append(int(utilityString[idx]))
            pmus.append(int(pmuString[idx]))
        return Transaction(items, utilities, transactionUtility, pmus)

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


class TKSHUIM(utilityPatterns):
    """
      Top K Spatial High Utility ItemSet Mining (TKSHUIM) aims to discover Top-K Spatial High Utility Itemsets
      (TKSHUIs) in a spatioTemporal database
    Reference:
    ---------


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
        k : int
            The user given k value
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
        itemsToKeep: list
            keep only the promising items ie items having twu >= minUtil
        itemsToExplore: list
            keep items that subtreeUtility grater than minUtil

    Methods :
    -------
        startMine()
                Mining process will start from here
        getPatterns()
                Complete set of patterns will be retrieved with this function
        savePatterns(oFile)
                Complete set of patterns will be loaded in to a output file
        getPatternsAsDataFrame()
                Complete set of patterns will be loaded in to a dataframe
        getMemoryUSS()
                Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
                Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
               Total amount of runtime taken by the mining process will be retrieved from this function
        calculateNeighbourIntersection(self, prefixLength)
               A method to return common Neighbours of items
        backtrackingEFIM(transactionsOfP, itemsToKeep, itemsToExplore, prefixLength)
               A method to mine the TKSHUIs Recursively
        useUtilityBinArraysToCalculateUpperBounds(transactionsPe, j, itemsToKeep, neighbourhoodList)
               A method to  calculate the sub-tree utility and local utility of all items that can extend itemSet P and e
        output(tempPosition, utility)
               A method ave a high-utility itemSet to file or memory depending on what the user chose
        is_equal(transaction1, transaction2)
               A method to Check if two transaction are identical
        intersection(lst1, lst2)
               A method that return the intersection of 2 list
        useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(dataset)
              Scan the initial database to calculate the subtree utility of each items using a utility-bin array
        sortDatabase(self, transactions)
              A Method to sort transaction in the order of PMU
        sort_transaction(self, trans1, trans2)
              A Method to sort transaction in the order of PMU
        useUtilityBinArrayToCalculateLocalUtilityFirstTime(self, dataset)
             A method to scan the database using utility bin array to calculate the pmus                   

    Executing the code on terminal :
    -------
        Format: python3 TKSHUIM.py <inputFile> <outputFile> <Neighbours> <k> <sep>
        Examples: python3 TKSHUIM.py sampleTDB.txt output.txt sampleN.txt 35  (it will consider "\t" as separator)
                  python3 TKSHUIM.py sampleTDB.txt output.txt sampleN.txt 35 , (it will consider "," as separator)

    Sample run of importing the code:
    -------------------------------
        
        from PAMI.highUtilitySpatialPattern.topk import TKSHUIM as alg

        obj=alg.TKSHUIM("input.txt","Neighbours.txt",35)

        obj.startMine()

        Patterns = obj.getPatterns()

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
    candidateCount = 0
    utilityBinArrayLU = {}
    utilityBinArraySU = {}
    oldNamesToNewNames = {}
    newNamesToOldNames = {}
    strToint = {}
    intTostr = {}
    Neighbours = {}
    temp = [0] * 5000
    maxMemory = 0
    startTime = float()
    endTime = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    nFile = " "
    sep = "\t"
    minUtil = 0
    memoryUSS = float()
    memoryRSS = float()
    heapList = []

    def __init__(self, iFile, nFile, k, sep="\t"):
        super().__init__(iFile, nFile, k, sep)

    def startMine(self):
        self.startTime = time.time()
        self.finalPatterns = {}
        self.dataset = Dataset(self.iFile, self.sep)
        with open(self.nFile, 'r') as o:
            lines = o.readlines()
            for line in lines:
                line = line.split("\n")[0]
                line_split = line.split(self.sep)
                item = self.dataset.strToint.get(line_split[0])
                lst = []
                for i in range(1, len(line_split)):
                    lst.append(self.dataset.strToint.get(line_split[i]))
                self.Neighbours[item] = lst
        o.close()
        InitialMemory = psutil.virtual_memory()[3]
        self.useUtilityBinArrayToCalculateLocalUtilityFirstTime(self.dataset)
        itemsToKeep = []
        for key in self.utilityBinArrayLU.keys():
            if self.utilityBinArrayLU[key] >= self.minUtil:
                itemsToKeep.append(key)
        itemsToKeep = sorted(itemsToKeep, key=lambda x: self.utilityBinArrayLU[x])
        currentName = 1
        for idx, item in enumerate(itemsToKeep):
            self.oldNamesToNewNames[item] = currentName
            self.newNamesToOldNames[currentName] = item
            itemsToKeep[idx] = currentName
            currentName += 1
        for transaction in self.dataset.getTransactions():
            transaction.removeUnpromisingItems(self.oldNamesToNewNames)
        self.sortDatabase(self.dataset.getTransactions())
        emptyTransactionCount = 0
        for transaction in self.dataset.getTransactions():
            if len(transaction.getItems()) == 0:
                emptyTransactionCount += 1
        self.dataset.transactions = self.dataset.transactions[emptyTransactionCount:]
        self.useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(self.dataset)
        self.heapList = []
        itemsToExplore = []
        for item in itemsToKeep:
            if self.utilityBinArraySU[item] >= self.minUtil:
                itemsToExplore.append(item)
        commonitems = []
        for i in range(self.dataset.maxItem):
            commonitems.append(i)
        self.backtrackingEFIM(self.dataset.getTransactions(), itemsToKeep, itemsToExplore, 0)
        finalMemory = psutil.virtual_memory()[3]
        memory = (finalMemory - InitialMemory) / 10000
        if memory > self.maxMemory:
            self.maxMemory = memory
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = float()
        self.memoryRSS = float()
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        for item in self.heapList:
            self.finalPatterns[item[1]] = item[0]
        print('TOP-K mining process is completed by TKSHUIM')

    def backtrackingEFIM(self, transactionsOfP, itemsToKeep, itemsToExplore, prefixLength):
        """
            A method to mine the TKSHUIs Recursively

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
        self.candidateCount += len(itemsToExplore)
        for idx, e in enumerate(itemsToExplore):
            initialMemory = psutil.virtual_memory()[3]
            transactionsPe = []
            utilityPe = 0
            previousTransaction = transactionsOfP[0]
            consecutiveMergeCount = 0
            for transaction in transactionsOfP:
                items = transaction.getItems()
                if e in items:
                    positionE = items.index(e)
                    if transaction.getLastPosition() == positionE:
                        utilityPe += transaction.getUtilities()[positionE] + transaction.prefixUtility
                    else:
                        projectedTransaction = transaction.projectTransaction(positionE)
                        utilityPe += projectedTransaction.prefixUtility
                        if previousTransaction == transactionsOfP[0]:
                            previousTransaction = projectedTransaction
                        elif self.is_equal(projectedTransaction, previousTransaction):
                            if consecutiveMergeCount == 0:
                                items = previousTransaction.items[previousTransaction.offset:]
                                utilities = previousTransaction.utilities[previousTransaction.offset:]
                                itemsCount = len(items)
                                positionPrevious = 0
                                positionProjection = projectedTransaction.offset
                                while positionPrevious < itemsCount:
                                    utilities[positionPrevious] += projectedTransaction.utilities[positionProjection]
                                    positionPrevious += 1
                                    positionProjection += 1
                                previousTransaction.prefixUtility += projectedTransaction.prefixUtility
                                sumUtilities = previousTransaction.prefixUtility
                                previousTransaction = Transaction(items, utilities, previousTransaction.transactionUtility + projectedTransaction.transactionUtility)
                                previousTransaction.prefixUtility = sumUtilities
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
                            consecutiveMergeCount += 1
                        else:
                            transactionsPe.append(previousTransaction)
                            previousTransaction = projectedTransaction
                            consecutiveMergeCount = 0
                    transaction.offset = positionE
            if previousTransaction != transactionsOfP[0]:
                transactionsPe.append(previousTransaction)
            self.temp[prefixLength] = self.newNamesToOldNames[e]
            if utilityPe >= self.minUtil:
                self.output(prefixLength, utilityPe)
            neighbourhoodList = self.calculateNeighbourIntersection(prefixLength)
            self.useUtilityBinArraysToCalculateUpperBounds(transactionsPe, idx, itemsToKeep, neighbourhoodList)
            newItemsToKeep = []
            newItemsToExplore = []
            for l in range(idx + 1, len(itemsToKeep)):
                itemK = itemsToKeep[l]
                if self.utilityBinArraySU[itemK] >= self.minUtil:
                    if itemK in neighbourhoodList:
                        newItemsToExplore.append(itemK)
                        newItemsToKeep.append(itemK)
                elif self.utilityBinArrayLU[itemK] >= self.minUtil:
                    if itemK in neighbourhoodList:
                        newItemsToKeep.append(itemK)
            self.backtrackingEFIM(transactionsPe, newItemsToKeep, newItemsToExplore, prefixLength + 1)
            finalMemory = psutil.virtual_memory()[3]
            memory = (finalMemory - initialMemory) / 10000
            if self.maxMemory < memory:
                self.maxMemory = memory

    def useUtilityBinArraysToCalculateUpperBounds(self, transactionsPe, j, itemsToKeep, neighbourhoodList):
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
            self.utilityBinArrayLU[item] = 0
            self.utilityBinArraySU[item] = 0
        for transaction in transactionsPe:
            length = len(transaction.getItems())
            i = length - 1
            while i >= transaction.offset:
                item = transaction.getItems()[i]
                if item in itemsToKeep:
                    remainingUtility = 0
                    if self.newNamesToOldNames[item] in self.Neighbours:
                        item_neighbours = self.Neighbours[self.newNamesToOldNames[item]]
                        for k in range(i, length):
                            transaction_item = transaction.getItems()[k]
                            if self.newNamesToOldNames[transaction_item] in item_neighbours and transaction_item in neighbourhoodList:
                                remainingUtility += transaction.getUtilities()[k]

                    remainingUtility += transaction.getUtilities()[i]
                    self.utilityBinArraySU[item] += remainingUtility + transaction.prefixUtility
                    self.utilityBinArrayLU[item] += transaction.transactionUtility + transaction.prefixUtility
                i -= 1

    def calculateNeighbourIntersection(self, prefixLength):
        """
            A method to find common Neighbours
            Attributes:
            ----------
                :param prefixLength: the prefix itemSet
                :type prefixLength:int

        """
        intersectionList = self.Neighbours.get(self.temp[0])
        for i in range(1, prefixLength+1):
            intersectionList = self.intersection(self.Neighbours[self.temp[i]], intersectionList)
        finalIntersectionList = []
        if intersectionList is None:
            return finalIntersectionList
        for item in intersectionList:
            if item in self.oldNamesToNewNames:
                finalIntersectionList.append(self.oldNamesToNewNames[item])
        return finalIntersectionList
    
    def output(self, tempPosition, utility):
        """
         A method save all high-utility itemSet to file or memory depending on what the user chose

         Attributes:
         ----------
         :param tempPosition: position of last item
         :type tempPosition : int 
         :param utility: total utility of itemSet
         :type utility: int
        """
        s1 = ""
        for i in range(0, tempPosition+1):
            s1 += self.dataset.intTostr.get((self.temp[i]))
            if i != tempPosition:
                s1 += " "
        self.additemset(s1, utility)

    def is_equal(self, transaction1, transaction2):
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
    
    def intersection(self, lst1, lst2):
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

    def useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(self, dataset):
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
                if item not in self.utilityBinArraySU:
                    self.utilityBinArraySU[item] = 0
                if self.newNamesToOldNames[item] not in self.Neighbours:
                    self.utilityBinArraySU[item] += utilities[idx]
                    continue
                i = idx + 1
                sumSu = utilities[idx]
                while i < len(items):
                    if self.newNamesToOldNames[items[i]] in self.Neighbours[self.newNamesToOldNames[item]]:
                        sumSu += utilities[i]
                    i += 1
                self.utilityBinArraySU[item] += sumSu

    def sortDatabase(self, transactions):
        """
            A Method to sort transaction in the order of PMU

            Attributes:
            ----------
            :param transactions: transaction of items
            :type transactions: Transaction 
            :return: sorted transaction
            :rtype: Transaction
        """
        cmp_items = cmp_to_key(self.sort_transaction)
        transactions.sort(key=cmp_items)

    def sort_transaction(self, trans1, trans2):
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

    def useUtilityBinArrayToCalculateLocalUtilityFirstTime(self, dataset):
        """
            A method to scan the database using utility bin array to calculate the pmus
            Attributes:
            ----------
            :param dataset: the transaction database
            :type dataset: database

        """
        utilityMatrix = defaultdict(lambda: defaultdict(int))
        for transaction in dataset.getTransactions():
            for idx, item in enumerate(transaction.getItems()):
                if item in self.utilityBinArrayLU:
                    self.utilityBinArrayLU[item] += transaction.getPmus()[idx]
                else:
                    self.utilityBinArrayLU[item] = transaction.getPmus()[idx]
                utilityMatrix[item][item] += transaction.getUtilities()[idx]
                if item in self.Neighbours:
                    neighbors = self.Neighbours[item]
                    utility = transaction.getUtilities()[idx]
                    for i, itemj in enumerate(transaction.getItems()):
                        if (itemj != item) and (itemj in neighbors):
                            utilityMatrix[item][itemj] += (utility + transaction.getUtilities()[i])

        for item in utilityMatrix.keys():
            for itemj in utilityMatrix[item].keys():
                if itemj >= item:
                    val = utilityMatrix[item][itemj]
                    if val != 0 and val > self.minUtil:
                        if itemj == item:
                            itemset = str(item)
                        else:
                            itemset = str(item) + str(itemj)
                        self.additemset(itemset, val)

    def additemset(self, itemset, utility):
        """
        adds the itemset to the priority queue
        """
        heapq.heappush(self.heapList, (utility, itemset))
        if len(self.heapList) > self.k:
            while len(self.heapList) > self.k:
                heapq.heappop(self.heapList)
                if len(self.heapList) == 0:
                    break
            self.minUtil = heapq.nsmallest(1, self.heapList)[0][0]

    def getPatternsAsDataFrame(self):
        """Storing final patterns in a dataframe

        :return: returning patterns in a dataframe
        :rtype: pd.DataFrame
        """
        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Utility'])

        return dataFrame
    
    def getPatterns(self):
        """ Function to send the set of patterns after completion of the mining process

        :return: returning patterns
        :rtype: dict
        """
        return self.finalPatterns

    def savePatterns(self, outFile):
        """Complete set of patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            patternsAndSupport = str(x) + " : " + str(y)
            writer.write("%s \n" % patternsAndSupport)

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
        return self.endTime-self.startTime


if __name__ == '__main__':
    ap = str()
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:
            ap = TKSHUIM(sys.argv[1], sys.argv[3], int(sys.argv[4]), sys.argv[5])
        if len(sys.argv) == 5:
            ap = TKSHUIM(sys.argv[1], sys.argv[3], int(sys.argv[4]))
        ap.startMine()
        ap.savePatterns(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in seconds:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
