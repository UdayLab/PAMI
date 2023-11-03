import random as _rd
import sys as _sys
import pandas as pd
import numpy as np

class createGeoreferentialTemporalDatabase:


    def __init__(self, transactions: int, N: int, avgTransaction: int, minimumXvalue: float, maximumXvalue: float, minimumYvalue: float, maximumYvalue: float) -> None:
        self._totalTransactions = transactions
        self._N = N
        self._avgTransactionLength = avgTransaction
        self._minX = minimumXvalue
        self._maxX = maximumXvalue
        self._minY = minimumYvalue
        self._maxY = maximumYvalue
        self._items = set()
        self._data = []

    def generate_distinct_items(self):
        while len(self._items) < self._N:
            x = _rd.uniform(self._minX, self._maxX)
            y = _rd.uniform(self._minY, self._maxY)
            item = f"{x},{y}"
            self._items.add(item)

    def generate_data(self):
        self.generate_distinct_items()

        for _ in range(self._totalTransactions):
            length = _rd.randint(1, self._avgTransactionLength + 20)
            transaction = _rd.sample(self._items, k=length)
            self._data.append(transaction)

    def save_to_csv(self, output_file: str) -> None:
        data = {'Transaction': self._data}
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

    def createGeoreferentialTemporalDatabase(self, outputFile: str) -> None:
        self.generate_data()
        self.save_to_csv(outputFile)

if __name__ == "__main__":
    if len(_sys.argv) == 8:
        transactions = int(_sys.argv[1])
        N = int(_sys.argv[2])
        avgTransaction = int(_sys.argv[3])
        minX = float(_sys.argv[4])
        maxX = float(_sys.argv[5])
        minY = float(_sys.argv[6])
        maxY = float(_sys.argv[7])
        _ap = createGeoreferentialTemporalDatabase(transactions, N, avgTransaction, minX, maxX, minY, maxY)
        _ap.createGeoreferentialTemporalDatabase("geo_temp.txt")
    else:
        print("Error! Please provide all seven command-line arguments.")
