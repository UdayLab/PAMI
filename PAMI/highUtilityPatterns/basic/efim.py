
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
import time
import os
import psutil
import pandas as pd
from functools import cmp_to_key
from PAMI.highUtilityPatterns.basic.abstract import *

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
        getLastPosition():
            return last position in a transaction
        removeUnpromisingItems():
            A method to remove items with low Utility than minUtil
        insertionSort():
            A method to sort all items in the transaction
    """
    offset = 0
    prefixUtility = 0

    def __init__(self, items, utilities, transactionUtility):
        self.items = items
        self.utilities = utilities
        self.transactionUtility = transactionUtility

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

    def removeUnpromisingItems(self, oldNamesToNewNames):
        """
            A method to remove items with low Utility than minUtil

            Parameters:
            -----------
            :param oldNamesToNewNames: A map represet old namses to new names
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
    def __init__(self, datasetpath,sep):
        self.strToint={}
        self.intTostr={}
        self.cnt=1
        self.sep=sep
        print(self.sep)
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
        items = []
        utilities = []
        for idx, item in enumerate(itemsString):
            if (self.strToint).get(item) is None:
                self.strToint[item]=self.cnt
                self.intTostr[self.cnt]=item
                self.cnt+=1
            item_int =self.strToint.get(item)
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

class efim(utilityPatterns):
    """
    efim is one of the fastest algorithm to mine High Utility Itemsets from transactional databases.
    
    Reference:
    ---------
        Zida, S., Fournier-Viger, P., Lin, J.CW. et al. efim: a fast and memory efficient algorithm for 
        high-utility itemset mining. Knowl Inf Syst 51, 595–625 (2017). https://doi.org/10.1007/s10115-016-0986-0
    
    Attributes:
    -----------
        iFile : file
            Name of the input file to mine complete set of frequent patterns
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
        highUtilityItemsets: map
            set of high utility itemsets
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
        maxMemory: float
        Maximum memory used by this program for runnning
        patternCount: int
            Number of SHUI's
        itemsToKeep: list
            keep only the promising items ie items having twu >= minUtil
        itemsToExplore: list
            keep items that subtreeUtility greter than minUtil

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
        backtrackingefim(transactionsOfP, itemsToKeep, itemsToExplore, prefixLength)
               A method to mine the SHUIs Recursively
        useUtilityBinArraysToCalculateUpperBounds(transactionsPe, j, itemsToKeep)
               A method to  calculate the sub-tree utility and local utility of all items that can extend itemset P and e
        output(tempPosition, utility)
               A method ave a high-utility itemset to file or memory depending on what the user chose
        is_equal(transaction1, transaction2)
               A method to Check if two transaction are identical
        useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(dataset)
              Scan the initial database to calculate the subtree utility of each items using a utility-bin array
        sortDatabase(self, transactions)
              A Method to sort transaction in the order of PMU
        sort_transaction(self, trans1, trans2)
              A Method to sort transaction in the order of PMU
        useUtilityBinArrayToCalculateLocalUtilityFirstTime(self, dataset)
             A method to scan the database using utility bin array to caluclate the pmus                   

    Executing the code on terminal :
    -------
        Format: python3 efim <inputFile> <outputFile> <Neighbours> <minUtil> <sep>
        Examples: python3 efim sampleTDB.txt output.txt sampleN.txt 35  (it will consider "\t" as separator)
                  python3 efim sampleTDB.txt output.txt sampleN.txt 35 , (it will consider "," as separator)

    Sample run of importing the code:
    -------------------------------
        
        import efim as alg

        obj=alg.efim("input.txt",35)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of high utility Patterns:", len(frequentPatterns))

        obj.storePatternsInFile("output")

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

    highUtilityItemsets=[]
    candidateCount = 0
    utilityBinArrayLU = {}
    utilityBinArraySU = {}
    oldNamesToNewNames = {}
    newNamesToOldNames = {}
    strToint={}
    intTostr={}
    Neighbours = {}
    temp = [0]*5000
    maxMemory = 0
    startTime = float()
    endTime = float()
    minSup = str()
    maxPer = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    nFile=" "
    sep="\t"
    minUtil=0
    memoryUSS = float()
    memoryRSS = float()

    def __init__(self,iFile,minUtil,sep="\t"):
        super().__init__(iFile,minUtil,sep)

    def startMine(self):
        self.startTime= time.time()
        self.patternCount=0
        self.dataset = Dataset(self.iFile,self.sep)
        f = open(self.oFile, 'w')
        self.useUtilityBinArrayToCalculateLocalUtilityFirstTime(self.dataset)
        minUtil=int(self.minUtil)
        itemsToKeep = []
        for key in self.utilityBinArrayLU.keys():
            if self.utilityBinArrayLU[key] >= minUtil:
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
        itemsToExplore = []
        for item in itemsToKeep:
            if self.utilityBinArraySU[item] >= minUtil:
                itemsToExplore.append(item)
        self.backtrackingefim(self.dataset.getTransactions(), itemsToKeep, itemsToExplore, 0)
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss

    def backtrackingefim(self, transactionsOfP, itemsToKeep, itemsToExplore, prefixLength):
        """
            A method to mine the SHUIs Recursively

            Attributes:
            ----------
            :param transactionOfP: the list of transactions containing the current prefix P
            :type transactionOfP: list 
            :param itemsToKeep: the list of secondary items in the p-projected database
            :type itemsToKeep: list
            :param itemsToExplore: the list of primary items in the p-projected database
            :type itemsToExplore: list
            :param prefixLength: current prefixLength
            :type prefixLength: int
        """
        self.candidateCount += len(itemsToExplore)
        for idx, e in enumerate(itemsToExplore):
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
            self.useUtilityBinArraysToCalculateUpperBounds(transactionsPe, idx, itemsToKeep)
            newItemsToKeep = []
            newItemsToExplore = []
            for l in range(idx + 1, len(itemsToKeep)):
                itemk = itemsToKeep[l]
                if self.utilityBinArraySU[itemk] >= self.minUtil:
                    newItemsToExplore.append(itemk)
                    newItemsToKeep.append(itemk)
                elif self.utilityBinArrayLU[itemk] >= self.minUtil:
                    newItemsToKeep.append(itemk)
            self.backtrackingefim(transactionsPe, newItemsToKeep, newItemsToExplore, prefixLength + 1)

    def useUtilityBinArraysToCalculateUpperBounds(self, transactionsPe, j, itemsToKeep):
        """
            A method to  calculate the sub-tree utility and local utility of all items that can extend itemset P U {e}

            Attributes:
            -----------
            :param transactionPe: transactions the projected database for P U {e}
            :type transaction: list
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
            sumRemainingUtility = 0
            i = len(transaction.getItems()) - 1
            while i >= transaction.offset:
                item = transaction.getItems()[i]
                if item in itemsToKeep:
                    sumRemainingUtility += transaction.getUtilities()[i]
                    self.utilityBinArraySU[item] += sumRemainingUtility + transaction.prefixUtility
                    self.utilityBinArrayLU[item] += transaction.transactionUtility + transaction.prefixUtility
                i -= 1

    def output(self, tempPosition, utility):
        """
         Method to print high utility items

         Attiributes:
         ----------
         :param tempPosition: postion of last item 
         :type tempPosition : int 
         :param utility: total utility of itemset
         :type utility: int
        """
        self.patternCount += 1
        s1=""
        for i in range(0, tempPosition+1):
            s1+=self.dataset.intTostr.get((self.temp[i]))
            if i != tempPosition:
                s1+=" "
        self.finalPatterns[s1]=str(utility)

    def is_equal(self, transaction1, transaction2):
        """
         A method to Check if two transaction are identical

         Attiributes:
         ----------
         :param  transaction1: the first transaction
         :type  transaction1: Trasaction
         :param  transaction2:    the second transaction
         :type  transaction2: Transaction
         :rteurn : whether both are identical or not
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
        :type dataset: list
        """
        for transaction in dataset.getTransactions():
            sumSU = 0
            i = len(transaction.getItems()) - 1
            while i >= 0:
                item = transaction.getItems()[i]
                sumSU += transaction.getUtilities()[i]
                if item in self.utilityBinArraySU.keys():
                    self.utilityBinArraySU[item] += sumSU
                else:
                    self.utilityBinArraySU[item] = sumSU
                i -= 1

    def sortDatabase(self, transactions):
        """
            A Method to sort transaction in the order of PMU

            Attributes:
            ----------
            :param transaction: transaction of items
            :type transaction: Transaction 
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
            A method to scan the database using utility bin array to caluclate the pmus
            Attributes:
            ----------
            :param dataset: the transaction database
            :type dataset: database

        """
        for transaction in dataset.getTransactions():
            for item in transaction.getItems():
                if item in self.utilityBinArrayLU:
                    self.utilityBinArrayLU[item] += transaction.transactionUtility
                else:
                    self.utilityBinArrayLU[item] = transaction.transactionUtility

    def getPatternsInDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
            """
        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support'])

        return dataFrame
    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns

    def storePatternsInFile(self, outFile):
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
    if len(sys.argv)==4 or len(sys.argv)==5:
        if len(sys.argv)==5: #includes separator
           ap=efim(sys.argv[1],int(sys.argv[3]),sys.argv[4])
        if len(sys.argv)==4: #takes "\t" as a separator
           ap=efim(sys.argv[1],int(sys.argv[3]))
        ap.startMine()
        patterns = ap.getPatterns()
        print("Total number of Spatial High Utility Patterns:", len(patterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in seconds:", run)
    else:
         print("Error! The number of input parameters do not match the total number of parameters provided")
