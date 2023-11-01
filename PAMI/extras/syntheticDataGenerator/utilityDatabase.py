import random as _rd
import sys as _sys

class createSyntheticUtility:
    
    def __init__(self, transactions: int, items: int, maxUtilRange: int, avgTransaction: int) -> None:
        self._totalTransactions = transactions
        self._noOfItems = items
        self._maxUtilRange = maxUtilRange
        self._avgTransactionLength = avgTransaction
        self._data = []

    def generate(self) -> None:
        for _ in range(self._totalTransactions):
            length = _rd.randint(1, self._avgTransactionLength + 20)
            transaction = []
            utilities = []
            for _ in range(length):
                item = _rd.randint(1, self._noOfItems)
                utility = _rd.randint(1, self._maxUtilRange) 
                transaction.append(item)
                utilities.append(utility)
            total_utility = sum(utilities)
            self._data.append((transaction, utilities, total_utility))

    def save_to_csv(self, output_file: str) -> None:
        with open(output_file, 'w') as writer:
            for transaction, utilities, total_utility in self._data:
                transaction_str = '\t'.join(map(str, transaction))
                utility_str = '\t'.join(map(str, utilities))
                writer.write(f"{transaction_str}:{total_utility}:\n")
                writer.write(f"{utility_str}\n")

    def createUtilityDatabase(self, outputFile: str) -> None:
        self.generate()
        self.save_to_csv(outputFile)

if __name__ == "__main__":
    _ap = createSyntheticUtility(100000, 870, 100, 10)
    _ap.createUtilityDatabase("T10_util.txt")
else:
    print("Error! The number of input parameters does not match the total number of parameters provided")
