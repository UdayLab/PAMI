from Transaction import Transaction


class Dataset:
    transactions = []
    maxItem = 0

    def __init__(self, datasetpath):
        with open(datasetpath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.transactions.append(self.createTransaction(line))
        print('Transaction Count :' + str(len(self.transactions)))
        f.close()

    def createTransaction(self, line):
        trans_list = line.strip().split(':')
        transactionUtility = int(trans_list[1])
        itemsString = trans_list[0].strip().split(' ')
        utilityString = trans_list[2].strip().split(' ')
        items = []
        utilities = []
        for idx, item in enumerate(itemsString):
            item_int = int(item)
            if item_int > self.maxItem:
                self.maxItem = item_int
            items.append(item_int)
            utilities.append(int(utilityString[idx]))
        return Transaction(items, utilities, transactionUtility)

    def getMaxItem(self):
        return self.maxItem

    def getTransactions(self):
        return self.transactions