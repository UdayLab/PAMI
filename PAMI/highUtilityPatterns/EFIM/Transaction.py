class Transaction:
    offset = 0
    prefixUtility = 0

    # constructor of a transaction
    def __init__(self, items, utilities, transactionUtility):
        self.items = items
        self.utilities = utilities
        self.transactionUtility = transactionUtility

    def projectTransaction(self, offsetE):
        new_transaction = Transaction(self.items, self.utilities, self.transactionUtility)
        utilityE = self.utilities[offsetE]
        new_transaction.prefixUtility = self.prefixUtility + utilityE
        new_transaction.transactionUtility = self.transactionUtility - utilityE
        for i in range(self.offset, offsetE):
            new_transaction.transactionUtility -= self.utilities[i]
        new_transaction.offset = offsetE + 1
        return new_transaction

    def getItems(self):
        return self.items

    def getUtilities(self):
        return self.utilities

    # get the last position in this transaction
    def getLastPosition(self):
        return len(self.items) - 1

    # This method removes unpromising items from the transaction and
    # at the same time rename items from old names to new names
    def removeUnpromisingItems(self, oldNamesToNewNames):
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