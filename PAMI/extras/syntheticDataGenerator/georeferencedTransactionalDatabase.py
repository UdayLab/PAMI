import random as _rd
import sys as _sys
import pandas as pd
import numpy as np

class GeoreferentialTransactionalDatabase:
    
    def __init__(self, transactions: int, items: int, avgTransaction: int) -> None:
        self._totalTransactions = transactions
        self._noOfItems = items
        self._avgTransactionLength = avgTransaction
        self._data = []

    def generate(self) -> None:
        items = []
        for _ in range(self._noOfItems):
            lat = _rd.uniform(-90, 90)
            lon = _rd.uniform(-180, 180)
            items.append((lat, lon))
        
        for _ in range(self._totalTransactions):
            length = _rd.randint(1, self._avgTransactionLength + 20)
            transaction = _rd.choices(items, k=length)
            self._data.append(transaction)

    def save(self, output_file: str) -> None:
        with open(output_file, 'w') as writer:
            for transaction in self._data:
                transaction_str = ' '.join(f'({lat} {lon})' for lat, lon in transaction)
                writer.write(f"{transaction_str}\n")

    def GeoreferentialTransactionalDatabase(self, outputFile: str) -> None:
        self.generate()
        self.save(outputFile)

if __name__ == "__main__":
    if len(_sys.argv) == 4:
        transactions = int(_sys.argv[1])
        items = int(_sys.argv[2])
        avgTransaction = int(_sys.argv[3])
        _ap = GeoreferentialTransactionalDatabase(transactions, items, avgTransaction)
        _ap.createGeoreferentialTransactionalDatabase("geo_trans.txt")
    else:
        print("Error! Please provide all three command-line arguments.")
      
