import numpy as np
import pandas as pd
import time
import os
import psutil


class UncertainTransactionalDatabase:
    """
    Generates an uncertain transactional database where each item in a transaction
    is associated with a random probability value.
    Each line is formatted as:
        item1 item2 ... itemk:prob1 prob2 ... probk
    """
    def __init__(self,
                 databaseSize: int,
                 avgItemsPerTransaction: float,
                 numItems: int,
                 sep: str = '\t',
                 itemDist: str = 'poisson'  # 'poisson' or 'uniform' distribution for item counts
                 ):
        self.databaseSize = databaseSize
        self.avgItems = avgItemsPerTransaction
        self.numItems = numItems
        self.sep = sep
        self.itemDist = itemDist
        self._proc = psutil.Process(os.getpid())

    def create(self):
        start = time.time()
        transactions = []

        for _ in range(self.databaseSize):
            # determine number of items in this transaction
            if self.itemDist == 'poisson':
                k = np.random.poisson(lam=self.avgItems)
            else:
                k = np.random.randint(1, max(2, int(2 * self.avgItems)))
            k = max(1, min(k, self.numItems))

            # sample unique item IDs
            items = np.random.choice(
                np.arange(1, self.numItems + 1), size=k, replace=False
            )
            items = list(map(str, items))

            # generate a random probability for each item
            probs = np.random.random(size=k)
            probs = [f"{p:.3f}" for p in probs]

            # format: "item1 item2 ... itemk:prob1 prob2 ... probk"
            txn_str = f"{self.sep.join(items)}:{self.sep.join(probs)}"
            transactions.append(txn_str)

        # build DataFrame
        self._df = pd.DataFrame({'transaction': transactions})

        end = time.time()
        self._runtime = end - start
        self._rss = self._proc.memory_info().rss
        self._uss = self._proc.memory_full_info().uss

    def save(self, filename: str):
        """Save each transaction line to a file (no header)."""
        with open(filename, 'w') as f:
            for txn in self._df['transaction']:
                f.write(txn + '\n')

    def getTransactions(self) -> pd.DataFrame:
        """Return the DataFrame of generated uncertain transactions."""
        return self._df

    def getRuntime(self) -> float:
        return self._runtime

    def getMemoryRSS(self) -> int:
        return self._rss

    def getMemoryUSS(self) -> int:
        return self._uss


# Example usage:
# obj = UncertainTransactionalDatabase(
#     databaseSize=100000,
#     avgItemsPerTransaction=10,
#     numItems=1000,
#     sep='\t',
#     itemDist='poisson'
# )
# obj.create()
# obj.save('uncertainTDB.csv')
# df = obj.getTransactions()
# print('Runtime:', obj.getRuntime())
# print('Memory (RSS):', obj.getMemoryRSS())
# print('Memory (USS):', obj.getMemoryUSS())
