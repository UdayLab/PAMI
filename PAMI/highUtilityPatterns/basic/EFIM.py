
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
from functools import cmp_to_key
from Transaction import Transaction
from Dataset import Dataset


class EFIM:
    """
    EFIM is one of the fastest algorithm to mine High Utility Itemsets from transactional databases.
    
    Reference:
    ---------
        Zida, S., Fournier-Viger, P., Lin, J.CW. et al. EFIM: a fast and memory efficient algorithm for 
        high-utility itemset mining. Knowl Inf Syst 51, 595–625 (2017). https://doi.org/10.1007/s10115-016-0986-0
    
    Methods
    -------
        runAlgorithm()
            The mining process starts.
        backtrackingEFIM()
            The recursive search procedure (mentioned in the paper) starts.
        useUtilityBinArraysToCalculateUpperBounds()
            Calculates the local utility and sub tree utility values.
        output()
            outputs the discovered itemset to the output File.
        is_equal()
            Check whether transactions are equal or not
        useUtilityBinArrayToCalculateSubtreeUtilityFirstTime()
            Calculates the sub tree utility values of single items present in the database.
        sortDatabase()
            Sorts the whole database
        sort_transaction()
            Comperator function between the transactions
        useUtilityBinArrayToCalculateLocalUtilityFirstTime()
            Calculates the local tree utility values of single items (TWU) present in the database.
        printStats()
            Print the details like # of HUIs, # of candidate itemsets, time taken, memory consumed.
    

    Executing the code on terminal:
    -------
        Format:
        ------
        python3 EFIM.py <inputFile> <outputFile> <minUtil>
        
        Examples:
        -------
        python3 EFIM.py sampleDB.txt patterns.txt 10
    
    Credits:
    -------
        @author pradeep pallikila
    """

    startTimestamp = 0
    endTimestamp = 0
    # minimum utility value given by the user
    minUtil = 0
    # number of candidate itemsets generated
    candidateCount = 0
    # an dictionary to hold the twu values of the items in database
    utilityBinArrayLU = {}
    # an dictionary to hold the subtree utility values of the items is database
    utilityBinArraySU = {}
    # an dictionary to store the new name corresponding to old name
    oldNamesToNewNames = {}
    # an dictionary to store the old name corresponding to new name
    newNamesToOldNames = {}
    # a temporary buffer
    temp = []
    for i in range(5000):
        temp.append(0)

    def __init__(self, inputPath, outputPath):
        self.patternCount = 0
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.f = open(outputPath, 'w')

    def runAlgorithm(self, minUtil):
        self.startTimestamp = time.time()
        dataset = Dataset(self.inputPath)
        self.minUtil = minUtil
        f = open(self.outputPath, 'w')
        # scan the database using utility bin array to caluclate
        self.useUtilityBinArrayToCalculateLocalUtilityFirstTime(dataset)
        # now we keep only the promising items ie items having twu >= minUtil
        itemsToKeep = []
        for key in self.utilityBinArrayLU.keys():
            if self.utilityBinArrayLU[key] >= minUtil:
                itemsToKeep.append(key)
        # sort the promising items according to increasing order of the twu
        itemsToKeep = sorted(itemsToKeep, key=lambda x: self.utilityBinArrayLU[x])
        # we will give the new names for all promising items starting from 1
        currentName = 1
        for idx, item in enumerate(itemsToKeep):
            self.oldNamesToNewNames[item] = currentName
            self.newNamesToOldNames[currentName] = item
            itemsToKeep[idx] = currentName
            currentName += 1
        # loop over every transaction in database to remove the unpromising items
        for transaction in dataset.getTransactions():
            transaction.removeUnpromisingItems(self.oldNamesToNewNames)
        # now we will sort the transactions according to proposed total order on transaction
        self.sortDatabase(dataset.getTransactions())
        # after removing the unimportant items from the database some items become empty
        # so remove those transactions
        emptyTransactionCount = 0
        for transaction in dataset.getTransactions():
            if len(transaction.getItems()) == 0:
                emptyTransactionCount += 1
        dataset.transactions = dataset.transactions[emptyTransactionCount:]
        # use utilitybinarraysu to caluclate the subtree utility of each item
        self.useUtilityBinArrayToCalculateSubtreeUtilityFirstTime(dataset)
        # prune the items which do not satisfy subtree utility conditions
        itemsToExplore = []
        for item in itemsToKeep:
            if self.utilityBinArraySU[item] >= minUtil:
                itemsToExplore.append(item)
        self.backtrackingEFIM(dataset.getTransactions(), itemsToKeep, itemsToExplore, 0)
        self.endTimestamp = time.time()

    def backtrackingEFIM(self, transactionsOfP, itemsToKeep, itemsToExplore, prefixLength):
        self.candidateCount += len(itemsToExplore)
        for idx, e in enumerate(itemsToExplore):
            # caluclate the transactions containing p U {e}
            # at the same time project transactions to keep what appears after e
            transactionsPe = []
            # variable to caluclate the utility of Pe
            utilityPe = 0
            # merging transactions
            previousTransaction = transactionsOfP[0]
            consecutiveMergeCount = 0
            for transaction in transactionsOfP:
                items = transaction.getItems()
                if e in items:
                    # if e was found in the transaction
                    positionE = items.index(e)
                    if transaction.getLastPosition() == positionE:
                        utilityPe += transaction.getUtilities()[positionE] + transaction.prefixUtility
                    else:
                        projectedTransaction = transaction.projectTransaction(positionE)
                        utilityPe += projectedTransaction.prefixUtility
                        if previousTransaction == transactionsOfP[0]:
                            # if it is the first transactoin
                            previousTransaction = projectedTransaction
                        elif self.is_equal(projectedTransaction, previousTransaction):
                            if consecutiveMergeCount == 0:
                                # if the first consecutive merge
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
            # caluclate the local utility and subtree utility
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
            self.backtrackingEFIM(transactionsPe, newItemsToKeep, newItemsToExplore, prefixLength + 1)

    def useUtilityBinArraysToCalculateUpperBounds(self, transactionsPe, j, itemsToKeep):
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
        self.patternCount += 1
        for i in range(0, tempPosition+1):
            self.f.write(str(self.temp[i]))
            if i != tempPosition:
                self.f.write(' ')
        self.f.write(' #UTIL: ')
        self.f.write(str(utility))
        self.f.write('\n')

    def is_equal(self, transaction1, transaction2):
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
        cmp_items = cmp_to_key(self.sort_transaction)
        transactions.sort(key=cmp_items)

    def sort_transaction(self, trans1, trans2):
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
        for transaction in dataset.getTransactions():
            for item in transaction.getItems():
                if item in self.utilityBinArrayLU:
                    self.utilityBinArrayLU[item] += transaction.transactionUtility
                else:
                    self.utilityBinArrayLU[item] = transaction.transactionUtility

    def printStats(self):
        print('EFIM STATS')
        print('Min Util = ' + str(self.minUtil))
        print('High Utility Itemsets count : ' + str(self.patternCount))
        print('Total Time : ' + str(self.endTimestamp - self.startTimestamp))
        print('Candidate Count : ' + str(self.candidateCount))


if __name__ == '__main__':
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    minUtil = int(sys.argv[3])
    q = EFIM(inputFile, outputFile)
    q.runAlgorithm(minUtil)
    process = psutil.Process(os.getpid())
    memoryUSS = (process.memory_full_info().uss) / 1024000
    memoryRSS = process.memory_info().rss / 1024000
    q.printStats()
    print("Total Memory in USS: " + str(memoryUSS) + " MB")
    print("Total Memory in RSS: " + str(memoryRSS) + " MB")