import random as _rd
import sys as _sys
import pandas as pd
import numpy as np

class georeferentialTemporalDatabase:

    
    def __init__(self, transactions: int, items: int, avgTransaction: int) -> None:
        self._totalTransactions = transactions
        self._noOfItems = items
        self._avgTransactionLength = avgTransaction
        self._data = []

    def generate(self) -> None:
        items = []
        for i in range(self._noOfItems):
            lat = _rd.uniform(-90, 90)
            lon = _rd.uniform(-180, 180)
            items.append((lat, lon))
        
        for _ in range(self._totalTransactions):
            length = _rd.randint(1, self._avgTransactionLength + 20)
            transaction = _rd.choices(items, k=length)
            self._data.append(transaction)

    def save(self, output_file: str) -> None:
        data = {'Transaction': self._data}
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

    def GeoreferentialTemporalDatabase(self, outputFile: str) -> None:
        self.generate_data()
        self.save_to_csv(outputFile)

if __name__ == "__main__":
    if len(_sys.argv) == 4:
        transactions = int(_sys.argv[1])
        items = int(_sys.argv[2])
        avgTransaction = int(_sys.argv[3])
        _ap = GeoreferentialTemporalDatabase(transactions, items, avgTransaction)
        _ap.createGeoreferentialTemporalDatabase("geo_temp.txt")
    else:
        print("Error! Please provide all three command-line arguments.")
