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


import sys
import validators
from urllib.request import urlopen
from PAMI.highUtilityFrequentPatterns.basic.abstract import *


class Transaction:
    """
        A class to store Transaction of a database

    Attributes:
    ----------
        items: list
            A list of items in transaction 
        utilities: list
            A list of utilities of items in transaction
        transactionUtility: int
            represent total sum of all utilities in the database
        prefixUtility:
            prefix Utility values of item
        offset:
            an offset pointer, used by projected transactions
        support:
            maintains the support of the transaction
    Methods:
    --------
        projectedTransaction(offsetE):
            A method to create new Transaction from existing starting from offsetE until the end
        getItems():
            return items in transaction
        getUtilities():
            return utilities in transaction
        getLastPosition():
            return last position in a transaction
        removeUnpromisingItems():
            A method to remove items which are having low values when compared with minUtil
        insertionSort():
            A method to sort all items in the transaction
        getSupport():
            returns the support of the transaction
    """
    offset = 0
    prefixUtility = 0
    support = 1

    def __init__(self, items, utilities, transactionUtility):
        self.items = items
        self.utilities = utilities
        self.transactionUtility = transactionUtility
        self.support = 1

    def projectTransaction(self, offsetE):
        """
            A method to create new Transaction from existing transaction starting from offsetE until the end

        Parameters:
        ----------
            :param offsetE: an offset over the original transaction for projecting the transaction
            :type offsetE: int
        """
        new_transaction = Transaction(self.items, self.utilities, self.transactionUtility)
        utilityE = self.utilities[offsetE]
        new_transaction.prefixUtility = self.prefixUtility + utilityE
        new_transaction.transactionUtility = self.transactionUtility - utilityE
        new_transaction.support = self.support
        for i in range(self.offset, offsetE):
            new_transaction.transactionUtility -= self.utilities[i]
        new_transaction.offset = offsetE + 1
        return new_transaction

    def getItems(self):
        """
            A method to return items in transaction
        """
        return self.items

    def getUtilities(self):
        """
            A method to return utilities in transaction
        """
        return self.utilities

    def getLastPosition(self):
        """
            A method to return last position in a transaction
        """

        return len(self.items) - 1

    def getSupport(self):
        """
            A method to return support in a transaction
        """

        return self.support

    def removeUnpromisingItems(self, oldNamesToNewNames):
        """
            A method to remove items which are not present in the map passed to the function

            Parameters:
            -----------
            :param oldNamesToNewNames: A map represent old names to new names
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
    
    def __init__(self, datasetPath, sep):
        self.strToInt = {}
        self.intToStr = {}
        self.cnt = 1
        self.sep = sep
        self.createItemSets(datasetPath)

    def createItemSets(self, datasetPath):
        self.Database = []
        if isinstance(datasetPath, pd.DataFrame):
            utilities, data, utilitySum = [], [], []
            if datasetPath.empty:
                print("its empty..")
            i = datasetPath.columns.values.tolist()
            if 'Transactions' in i:
                data = datasetPath['Transactions'].tolist()
            if 'utilities' in i:
                utilities = datasetPath['utilities'].tolist()
            if 'utilitySum' in i:
                utilitySum = datasetPath['utilitySum'].tolist()
            for k in range(len(data)):
                self.transactions.append(self.createTransaction(data[k], utilities[k], utilitySum[k]))
        if isinstance(datasetPath, str):
            if validators.url(datasetPath):
                data = urlopen(datasetPath)
                for line in data:
                    line = line.decode("utf-8")
                    trans_list = line.strip().split(':')
                    transactionUtility = int(trans_list[1])
                    itemsString = trans_list[0].strip().split(self.sep)
                    itemsString = [x for x in itemsString if x]
                    utilityString = trans_list[2].strip().split(self.sep)
                    utilityString = [x for x in utilityString if x]
                    self.transactions.append(self.createTransaction(itemsString, utilityString, transactionUtility))
            else:
                try:
                    with open(datasetPath, 'r', encoding='utf-8') as f:
                        for line in f:
                            trans_list = line.strip().split(':')
                            transactionUtility = int(trans_list[1])
                            itemsString = trans_list[0].strip().split(self.sep)
                            itemsString = [x for x in itemsString if x]
                            utilityString = trans_list[2].strip().split(self.sep)
                            utilityString = [x for x in utilityString if x]
                            self.transactions.append(self.createTransaction(itemsString, utilityString, transactionUtility))
                except IOError:
                    print("File Not Found")
                    quit()

    def createTransaction(self, items, utilities, utilitySum):
        """
            A method to create Transaction from dataset given
            
            Attributes:
            -----------
            :param items: represent a single line of database
            :type items: list
            :param utilities: represent a utilities of items
            :type utilities: list
            :param utilitySum: represent a the utilitySum
            :type items: int
            :return : Transaction
            :rtype: Transaction
        """
        transactionUtility = utilitySum
        itemsString = items
        utilityString = utilities
        items = []
        utilities = []
        for idx, item in enumerate(itemsString):
            if self.strToInt.get(item) is None:
                self.strToInt[item] = self.cnt
                self.intToStr[self.cnt] = item
                self.cnt += 1
            item_int = self.strToInt.get(item)
            if item_int > self.maxItem:
                self.maxItem = item_int
            items.append(item_int)
            utilities.append(int(utilityString[idx]))
        return Transaction(items, utilities, transactionUtility)

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


class HUFIM(utilityPatterns):
    """
    HUFIM (High Utility Frequent Itemset Miner) algorithm helps us to mine High Utility Frequent ItemSets (HUFIs) from transactional databases.


    Reference:
    ---------
        Kiran, R.U., Reddy, T.Y., Fournier-Viger, P., Toyoda, M., Reddy, P.K., & Kitsuregawa, M. (2019).
        Efficiently Finding High Utility-Frequent Itemsets Using Cutoff and Suffix Utility. PAKDD 2019.
        DOI: 10.1007/978-3-030-16145-3_15
    
    Attributes:
    -----------
        iFile : file
            Name of the input file to mine complete set of patterns
        oFile : file
            Name of the output file to store complete set of patterns
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        minUtil : int
            The user given minUtil value
        minSup : float
            The user given minSup value
        highUtilityFrequentItemSets: map
            set of high utility frequent itemSets
        candidateCount: int
             Number of candidates 
        utilityBinArrayLU: list
             A map to hold the local utility values of the items in database
        utilityBinArraySU: list
            A map to hold the subtree utility values of the items is database
        oldNamesToNewNames: list
            A map which contains old names, new names of items as key value pairs
        newNamesToOldNames: list
            A map which contains new names, old names of items as key value pairs
        singleItemSetsSupport: map
            A map which maps from single itemsets (items) to their support
        singleItemSetsUtility: map
            A map which maps from single itemsets (items) to their utilities
        maxMemory: float
            Maximum memory used by this program for running
        patternCount: int
            Number of RHUI's
        itemsToKeep: list
            keep only the promising items i.e items that can extend other items to form RHUIs
        itemsToExplore: list
            list of items that needs to be explored

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
        backTrackingHUFIM(transactionsOfP, itemsToKeep, itemsToExplore, prefixLength)
               A method to mine the RHUIs Recursively
        useUtilityBinArraysToCalculateUpperBounds(transactionsPe, j, itemsToKeep)
               A method to calculate the sub-tree utility and local utility of all items that can extend itemSet P and e
        output(tempPosition, utility)
               A method to output a relative-high-utility itemSet to file or memory depending on what the user chose
        is_equal(transaction1, transaction2)
               A method to Check if two transaction are identical
        useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(dataset)
              A method to calculate the sub tree utility values for single items
        sortDatabase(self, transactions)
              A Method to sort transaction
        sort_transaction(self, trans1, trans2)
              A Method to sort transaction
        useUtilityBinArrayToCalculateLocalUtilityFirstTime(self, dataset)
             A method to calculate local utility values for single itemSets

    Executing the code on terminal :
    -------
        Format:
            python3 HUFIM.py <inputFile> <outputFile> <minUtil> <sep>
        Examples:
            python3 HUFIM.py sampleTDB.txt output.txt 35 20 (it will consider "\t" as separator)

            python3 HUFIM.py sampleTDB.txt output.txt 35 20 , (it will consider "," as separator)

    Sample run of importing the code:
    -------------------------------
        
        from PAMI.highUtilityFrequentPatterns.basic import HUFIM as alg

        obj=alg.HUFIM("input.txt", 35, 20)

        obj.startMine()

        Patterns = obj.getPatterns()

        print("Total number of high utility frequent Patterns:", len(Patterns))

        obj.savePatterns("output")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
   
    Credits:
    -------
        The complete program was written by pradeep pallikila under the supervision of Professor Rage Uday Kiran.
     
    """

    highUtilityFrequentItemSets = []
    candidateCount = 0
    utilityBinArrayLU = {}
    utilityBinArraySU = {}
    oldNamesToNewNames = {}
    newNamesToOldNames = {}
    singleItemSetsSupport = {}
    singleItemSetsUtility = {}
    strToInt = {}
    intToStr = {}
    temp = [0]*5000
    patternCount = int()
    maxMemory = 0
    startTime = float()
    endTime = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    nFile = " "
    lno = 0
    sep = "\t"
    minUtil = 0
    minSup = 0
    memoryUSS = float()
    memoryRSS = float()

    def __init__(self, iFile, minUtil, minSup, sep="\t"):
        super().__init__(iFile, minUtil, minSup, sep)

    def convert(self, value):
        """
        To convert the given user specified value

        :param value: user specified value
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self.dataset.getTransactions()) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self.dataset.getTransactions()) * value)
            else:
                value = int(value)
        return value


    def startMine(self):
        self.startTime = time.time()
        self.dataset = Dataset(self.iFile, self.sep)
        self.singleItemSetsSupport = defaultdict(int)
        self.singleItemSetsUtility = defaultdict(int)
        self.useUtilityBinArrayToCalculateLocalUtilityFirstTime(self.dataset)
        self.minUtil = int(self.minUtil)
        self.minSup = self.convert(self.minSup)
        #print("######################################")
        #print("given minimum support is", self.minSup)
        #print("given minimum utility is", self.minUtil)
        itemsToKeep = []
        for key in self.utilityBinArrayLU.keys():
            if self.utilityBinArrayLU[key] >= self.minUtil and self.singleItemSetsSupport[key] >= self.minSup:
                itemsToKeep.append(key)
        # sorting items in decreasing order of utilities
        itemsToKeep = sorted(itemsToKeep, key=lambda x: self.singleItemSetsUtility[x], reverse=True)
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
        # calculating suffix utility values
        totalUtility = 0
        for item in itemsToKeep:
            totalUtility += self.singleItemSetsUtility[self.newNamesToOldNames[item]]
        # piItems
        piItems = []
        for item in itemsToKeep:
            if totalUtility >= self.minUtil:
                piItems.append(item)
                totalUtility -= self.singleItemSetsUtility[self.newNamesToOldNames[item]]
            else:
                break
        self.useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(self.dataset)
        itemsToExplore = []
        for item in piItems:
            if self.utilityBinArraySU[item] >= self.minUtil:
                itemsToExplore.append(item)
        self.backTrackingHUFIM(self.dataset.getTransactions(), itemsToKeep, itemsToExplore, 0)
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = float()
        self.memoryRSS = float()
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("High Utility Frequent patterns were generated successfully using HUFIM algorithm")

    def backTrackingHUFIM(self, transactionsOfP, itemsToKeep, itemsToExplore, prefixLength):
        """
            A method to mine the HUFIs Recursively

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
        # print("###############")
        # print("P is", [self.dataset.intToStr.get(x) for x in self.temp[:prefixLength]])
        # print("items to explore", [self.dataset.intToStr.get(x) for x in [self.newNamesToOldNames[y] for y  in itemsToExplore]])
        # print("items to keep", [self.dataset.intToStr.get(x) for x in [self.newNamesToOldNames[y] for y in itemsToKeep]])
        # print("--------------")
        self.candidateCount += len(itemsToExplore)
        for idx, e in enumerate(itemsToExplore):
            # print("exploring item", self.dataset.intToStr.get(self.newNamesToOldNames[e]))
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
                        elif self.is_equal(projectedTransaction, previousTransaction):
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
                                previousTransaction = Transaction(items, utilities, previousTransaction.transactionUtility + projectedTransaction.transactionUtility)
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
            # print("support is", supportPe)
            self.temp[prefixLength] = self.newNamesToOldNames[e]
            if (utilityPe >= self.minUtil) and (supportPe >= self.minSup):
                self.output(prefixLength, utilityPe, supportPe)
            if supportPe >= self.minSup:
                self.useUtilityBinArraysToCalculateUpperBounds(transactionsPe, idx, itemsToKeep)
                newItemsToKeep = []
                newItemsToExplore = []
                for l in range(idx + 1, len(itemsToKeep)):
                    itemK = itemsToKeep[l]
                    if self.utilityBinArraySU[itemK] >= self.minUtil:
                        newItemsToExplore.append(itemK)
                        newItemsToKeep.append(itemK)
                    elif self.utilityBinArrayLU[itemK] >= self.minUtil:
                        newItemsToKeep.append(itemK)
                if len(transactionsPe) != 0:
                    self.backTrackingHUFIM(transactionsPe, newItemsToKeep, newItemsToExplore, prefixLength + 1)

    def useUtilityBinArraysToCalculateUpperBounds(self, transactionsPe, j, itemsToKeep):
        """
            A method to  calculate the sub-tree utility and local utility of all items that can extend itemSet P U {e}

            Attributes:
            -----------
            :param transactionsPe: transactions the projected database for P U {e}
            :type transactionsPe: list or Dataset
            :param j:he position of j in the list of promising items
            :type j:int
            :param itemsToKeep :the list of promising items
            :type itemsToKeep: list or Dataset

        """
        for i in range(j + 1, len(itemsToKeep)):
            item = itemsToKeep[i]
            self.utilityBinArrayLU[item] = 0
            self.utilityBinArraySU[item] = 0
        for transaction in transactionsPe:
            sumRemainingUtility = 0
            i = len(transaction.getItems()) - 1
            while i >= transaction.offset:
                item = transaction.getItems()[i]
                if item in itemsToKeep:
                    sumRemainingUtility += transaction.getUtilities()[i]
                    self.utilityBinArraySU[item] += sumRemainingUtility + transaction.prefixUtility
                    self.utilityBinArrayLU[item] += transaction.transactionUtility + transaction.prefixUtility
                i -= 1

    def output(self, tempPosition, utility, support):
        """
         Method to print itemSets

         Attributes:
         ----------
         :param tempPosition: position of last item 
         :type tempPosition : int 
         :param utility: total utility of itemSet
         :type utility: int
         :param support: support of an itemSet
         :type support: int
        """
        self.patternCount += 1
        s1 = ""
        for i in range(0, tempPosition+1):
            s1 += self.dataset.intToStr.get((self.temp[i]))
            if i != tempPosition:
                s1 += " "
        self.finalPatterns[s1] = str(utility) + ":" + str(support)

    def is_equal(self, transaction1, transaction2):
        """
         A method to Check if two transaction are identical

         Attributes:
         ----------
         :param  transaction1: the first transaction
         :type  transaction1: Transaction
         :param  transaction2:    the second transaction
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

    def useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(self, dataset):
        """
        Scan the initial database to calculate the subtree utility of each items using a utility-bin array

        Attributes:
        ----------
        :param dataset: the transaction database
        :type dataset: Dataset
        """
        for transaction in dataset.getTransactions():
            sumSU = 0
            i = len(transaction.getItems()) - 1
            while i >= 0:
                item = transaction.getItems()[i]
                currentUtility = transaction.getUtilities()[i]
                sumSU += currentUtility
                if item in self.utilityBinArraySU.keys():
                    self.utilityBinArraySU[item] += sumSU
                else:
                    self.utilityBinArraySU[item] = sumSU
                i -= 1

    def sortDatabase(self, transactions):
        """
            A Method to sort transaction

            Attributes:
            ----------
            :param transactions: transaction of items
            :type transactions: list
            :return: sorted transactions
            :rtype: Transactions or list
        """
        cmp_items = functools.cmp_to_key(self.sort_transaction)
        transactions.sort(key=cmp_items)

    def sort_transaction(self, trans1, trans2):
        """
            A Method to sort transaction

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
            A method to calculate local utility of single itemSets
            Attributes:
            ----------
            :param dataset: the transaction database
            :type dataset: database

        """
        for transaction in dataset.getTransactions():
            for idx, item in enumerate(transaction.getItems()):
                self.singleItemSetsSupport[item] += 1
                self.singleItemSetsUtility[item] += transaction.getUtilities()[idx]
                if item in self.utilityBinArrayLU:
                    self.utilityBinArrayLU[item] += transaction.transactionUtility
                else:
                    self.utilityBinArrayLU[item] = transaction.transactionUtility

    def getPatternsAsDataFrame(self):
        """Storing final patterns in a dataframe

        :return: returning patterns in a dataframe
        :rtype: pd.DataFrame
            """
        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Utility:Support'])

        return dataFrame
    
    def getPatterns(self):
        """ Function to send the set of patterns after completion of the mining process

        :return: returning patterns
        :rtype: dict
        """
        return self.finalPatterns

    def savePatterns(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

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
        if len(sys.argv) == 6:    #includes separator
            ap = HUFIM(sys.argv[1], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
        if len(sys.argv) == 5:    #takes "\t" as a separator
            ap = HUFIM(sys.argv[1], int(sys.argv[3]), float(sys.argv[4]))
        ap.startMine()
        patterns = ap.getPatterns()
        print("Total number of High Utility Frequent Patterns:", ap.patternCount)
        print("Total number of Candidate Patterns:", ap.candidateCount)
        ap.savePatterns(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in seconds:", run)
        print("######################################")
    else:
        l = [0.004, 0.005, 0.006, 0.007, 0.008]
        for i in l:
            ap = HUFIM('/home/apiiit-rkv/Downloads/Reaserch/maximal/mushroom_utility_SPMF.txt',
                   i, 0.3, ' ')
            ap.startMine()
            patterns = ap.getPatterns()
            print("Total number of High Utility Frequent Patterns:", ap.patternCount)
            print("Total number of Candidate Patterns:", ap.candidateCount)
            ap.savePatterns('/home/apiiit-rkv/Downloads/output')
            memUSS = ap.getMemoryUSS()
            print("Total Memory in USS:", memUSS)
            memRSS = ap.getMemoryRSS()
            print("Total Memory in RSS", memRSS)
            run = ap.getRuntime()
            print("Total ExecutionTime in seconds:", run)
        print("Error! The number of input parameters do not match the total number of parameters provided")
