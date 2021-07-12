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

import time
import sys
from UPItem import UPItem
from UPTree import UPTree


class UPGrowth:

    """
    UP-Growth is two-phase algorithm to mine High Utility Itemsets from transactional databases.
    
    Reference:
    ---------
        Vincent S. Tseng, Cheng-Wei Wu, Bai-En Shie, and Philip S. Yu. 2010. UP-Growth: an efficient algorithm for high utility itemset mining. 
        In Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '10).
        Association for Computing Machinery, New York, NY, USA, 253–262. DOI:https://doi.org/10.1145/1835804.1835839
    
    Methods
    -------
        runAlgorithm()
            The mining process starts.
        printStats()
            Print the details like # of HUIs, # of PHUIs, time taken, memory consumed.
    

    Executing the code on terminal:
    -------
        Format:
        ------
        python3 UPGrowth.py <inputFile> <outputFile> <minUtil>
        
        Examples:
        -------
        python3 UPGrowth.py sampleDB.txt patterns.txt 100
    
    Credits:
    -------
        @author pradeep pallikila
    """

    # max memory used
    maxMemory = 0
    # start time stamp
    StartTime = 0
    # end time stamp
    EndTime = 0
    # minimum utility
    minUtility = 0
    # Total number of nodes generated while building the tree
    NumberOfNodes = 0
    # total number of nodes required to build the parent tree
    ParentNumberOfNodes = 0
    # a dictionary to store the minimum utility of item in the database
    MapItemToMinimumUtility = {}
    # a structure to store the phuis
    phuis = []
    # a dictionary to store the twu of each item in database
    MapItemToTwu = {}

    def __init__(self, inputFile, outputFile):
        self.inputFile = inputFile
        self.outputFile = outputFile

    def runAlgorithm(self, minUtility):
        self.StartTime = time.time()
        self.minUtility = minUtility
        # creating a global up tree
        tree = UPTree()
        with open(self.inputFile, 'r') as o:
            lines = o.readlines()
            for line in lines:
                transaction = line.strip().split(':')
                items = transaction[0].split(' ')
                transactionUtility = int(transaction[1])
                for item in items:
                    Item = int(item)
                    if Item in self.MapItemToTwu:
                        self.MapItemToTwu[Item] += transactionUtility
                    else:
                        self.MapItemToTwu[Item] = transactionUtility
            for line in lines:
                transaction = line.strip().split(':')
                items = transaction[0].split(' ')
                utilities = transaction[2].split(' ')
                remainingUtility = 0
                revisedTransaction = []
                for idx, item in enumerate(items):
                    Item = int(item)
                    utility = int(utilities[idx])
                    if self.MapItemToTwu[Item] >= minUtility:
                        element = UPItem(Item, utility)
                        revisedTransaction.append(element)
                        remainingUtility += utility
                        if Item in self.MapItemToMinimumUtility:
                            minItemUtil = self.MapItemToMinimumUtility[Item]
                            if utility < minItemUtil:
                                self.MapItemToMinimumUtility[Item] = utility
                        else:
                            self.MapItemToMinimumUtility[Item] = utility
                # sort revised transaction in descending order of twu
                revisedTransaction = sorted(revisedTransaction, key=lambda x: self.MapItemToTwu[x.name], reverse=True)
                # add the revised transaction in to the tree
                self.ParentNumberOfNodes += tree.addTransaction(revisedTransaction, remainingUtility)
        o.close()
        # print('printing maptominUtil')
        # print(self.MapItemToMinimumUtility)
        # we create the header table for the global up tree
        tree.createHeaderList(self.MapItemToTwu)
        # print('printing header table')
        # print(tree.headerList)
        # initially prefix itemset is a empty list
        alpha = []
        # mine the tree recursively using two stratergies DGU and DLN
        self.upgrowth(tree, alpha)
        # scan the database for finding utility of all PHUIS obtained by mining the up tree

        # before that sort the phuis generated by size for optimization
        self.phuis = sorted(self.phuis, key=lambda x: len(x))
        SortedPhuis = []
        for phui in self.phuis:
            itemset = sorted(phui, key=lambda x: x)
            SortedPhuis.append(itemset)
        self.phuis = SortedPhuis
        # calculate the exact utility
        with open(self.inputFile, 'r') as f:
            lines = f.readlines()
        f.close()

    def upgrowth(self, tree, alpha):
        for item in reversed(tree.headerList):
            # print('alpha and item')
            # print(alpha, item)
            # create a local tree
            localTree = self.createLocalTree(tree, item)
            # calculate the sum of item node utility
            # take node from bottom of the table
            node = tree.mapItemNodes[item]
            ItemTotalUtility = 0
            while node != -1:
                ItemTotalUtility += node.nodeUtility
                node = node.nodeLink
            if ItemTotalUtility >= self.minUtility:
                # create the itemset by appending this item to the prefix
                beta = alpha + [item]
                self.phuis.append(beta)
                # make a recursive call to the up growth procedure to explore
                # the other itemsets that are extensions of the current PHUI
                if len(localTree.headerList) > 0:
                    self.upgrowth(localTree, beta)

    def createLocalTree(self, tree, item):
        # construct the conditional pattern base for this item
        # It is a subdatabase which consists of set of prefix paths
        prefixPaths = []
        path = tree.mapItemNodes[item]
        # map to store the path utilities of local item in CPB
        itemPathUtility = {}
        while path != -1:
            nodeUtility = path.nodeUtility
            # if the path is not just the root node
            if path.parent != -1:
                # create a prefix path
                prefixPath = []
                # add this node
                prefixPath.append(path)
                # Note we add it just for completeness actually it should not be added
                # Recursively add all the parents of this node
                ParentNode = path.parent
                while ParentNode.itemId != -1:
                    prefixPath.append(ParentNode)
                    itemName = ParentNode.itemId
                    if itemName in itemPathUtility:
                        itemPathUtility[itemName] += nodeUtility
                    else:
                        itemPathUtility[itemName] = nodeUtility
                    ParentNode = ParentNode.parent
                prefixPaths.append(prefixPath)
            path = path.nodeLink
        # create a local tree
        localTree = UPTree()
        for prefixPath in prefixPaths:
            # the path utility of prefix path is utility of first node
            pathUtility = prefixPath[0].nodeUtility
            pathCount = prefixPath[0].count
            # reorganised prefix path will be stored in local path
            localPath = []
            for i in range(1, len(prefixPath)):
                node = prefixPath[i]
                if itemPathUtility[node.itemId] >= self.minUtility:
                    localPath.append(node.itemId)
                else:
                    # if the item in unpromising local item then we re calculate the path utility
                    pathUtility -= node.count * self.MapItemToMinimumUtility[node.itemId]
            # reorganize the local path in descending order of path utility
            localPath = sorted(localPath, key=lambda x: itemPathUtility[x], reverse=True)
            self.NumberOfNodes += localTree.addLocalTransaction(localPath, pathUtility, self.MapItemToMinimumUtility, pathCount)
        # create a header table for the tree item
        localTree.createHeaderList(itemPathUtility)
        return localTree

    def PrintStats(self):
        print('number of PHUIS are ' + str(len(self.phuis)))
        # print(self.phuis)


if __name__ == '__main__':
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    minUtil = int(sys.argv[3])
    q = UPGrowth(inputFile, outputFile)
    q.runAlgorithm(minUtil)
    q.PrintStats()