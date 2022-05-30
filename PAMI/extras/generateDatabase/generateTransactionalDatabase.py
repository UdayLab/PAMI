import random


class generateTransactionalDatabase:
    """
    generateTransactionalDatabase generates a transactional database

        Attributes:
        -----------
        numOfTransactions: int
            number of transactions
        maxNumOfDistinctItems: int
            maximum number of distinct items
        numOfItemsPerTransaction: int
            number of items per transaction
        outFileName: str
            output file name
        sep: str
            seperator in file, default is tab space

        Methods:
        --------
        getFileName()
            get output filename
    """
    def __init__(self, numOfTransactions, maxNumOfDistinctItems, numOfItemsPerTransaction, outFileName, sep='\t'):
        """

        :param numOfTransactions: number of transactions
        :type numOfTransactions: int
        :param maxNumOfDistinctItems: distinct items per transactions
        :type maxNumOfDistinctItems: int 
        :param numOfItemsPerTransaction: items per transaction
        :type numOfItemsPerTransaction: int
        :param outFileName: output filename
        :type outFileName: str
        :param sep: seperator
        :type sep: str
        """
        self.numOfTransactions = numOfTransactions
        self.maxNumOfDistinctItems = maxNumOfDistinctItems
        self.numOfItemsPerTransaction = numOfItemsPerTransaction
        self.outFileName = outFileName
        self.sep = sep

        # make outFile
        with open(self.outFileName, "w+") as outFile:
            # For the number of transactions to be generated
            for i in range(self.numOfTransactions):
                # This hashset will be used to remember which items have
                # already been added to this item set.
                alreadyAdded = set()
                # create an arraylist to store items from the item set that will be generated
                itemSet = list()
                # We randomly decide how many items will appear in this transaction
                randNumOfItems = random.randrange(self.maxNumOfDistinctItems) + 1
                # for the number of items that was decided above
                for j in range(randNumOfItems):
                    # we generate the item randomly and write it to disk
                    item = random.randrange(self.maxNumOfDistinctItems) + 1
                    # if we already added this item to this item set
                    # we choose another one
                    while item in alreadyAdded:
                        item = random.randrange(self.maxNumOfDistinctItems) + 1
                    alreadyAdded.add(item)
                    itemSet.append(item)
                # sort the item set
                itemSet.sort()
                # write the item set
                for j in itemSet:
                    outFile.write(str(j) + self.sep)
                outFile.write('\n')
        # close outFile
        outFile.close()



    def getFileName(self):
        """
        return output file name
        :return: output file name
        """
        return self.outFileName

if __name__ == '__main__':
    numOfTransactions = 500
    maxNumOfDistinctItems = 1000
    numOfItemsPerTransaction = 20
    outFileName = '/Users/Likhitha/Downloads/out.txt'

    tDG = generateTransactionalDatabase(numOfTransactions, maxNumOfDistinctItems, numOfItemsPerTransaction, outFileName)
