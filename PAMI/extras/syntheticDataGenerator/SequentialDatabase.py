import numpy as np
import pandas as pd
import time
import os
import psutil

class SequentialDatabase:
    def __init__(self,
                 databaseSize: int,
                 numItems: int,
                 avgItemsPerPatterns: int,
                 avgPatternsPerSequence: int,
                 seqSep: str = ':',
                 txnSep: str = '\t',
                 groupDist: str = 'poisson'  # 'poisson' or 'uniform'
                 ):
        self.databaseSize = databaseSize
        self.numItems = numItems
        self.avgItems = avgItemsPerPatterns
        self.avgTxns = avgPatternsPerSequence
        self.seqSep = seqSep
        self.txnSep = txnSep
        self.groupDist = groupDist
        self._proc = psutil.Process(os.getpid())

    def create(self):
        t0 = time.time()

        # 1) Generate each transaction as a fixed-size random sample of items
        txs = [
            list(np.random.choice(
                np.arange(1, self.numItems + 1),
                size=min(self.avgItems, self.numItems),
                replace=False
            ))
            for _ in range(self.databaseSize)
        ]

        # 2) Group transactions into sequences using chosen distribution
        seq_ids = []
        idx = 0
        seq_id = 1
        while idx < self.databaseSize:
            if self.groupDist == 'poisson':
                L = np.random.poisson(lam=self.avgTxns)
            else:
                L = np.random.randint(1, max(2, int(2 * self.avgTxns)))
            L = max(1, min(L, self.databaseSize - idx))
            seq_ids.extend([seq_id] * L)
            idx += L
            seq_id += 1

        # 3) Build sequence strings
        sequences = []
        total_sequences = seq_id - 1
        for sid in range(1, total_sequences + 1):
            group = [txs[i] for i, s in enumerate(seq_ids) if s == sid]
            seq_str = self.seqSep.join(
                self.txnSep.join(map(str, txn)) for txn in group
            )
            sequences.append(seq_str)

        # 4) Store into DataFrame
        self._df = pd.DataFrame({'sequence': sequences})

        t1 = time.time()
        self._runtime = t1 - t0
        self._rss = self._proc.memory_info().rss
        self._uss = self._proc.memory_full_info().uss

    def save(self, filename: str):
        # self._df.to_csv(filename, index=False)
        with open(filename, 'w') as f:
            for seq in self._df['sequence']:
                f.write(seq + '\n')


    def getTransactions(self) -> pd.DataFrame:
        return self._df

    def getRuntime(self) -> float:
        return self._runtime

    def getMemoryRSS(self) -> int:
        return self._rss

    def getMemoryUSS(self) -> int:
        return self._uss


if __name__ == '__main__':
# Example Usage:
    obj = SequentialDatabase(
        databaseSize=10000,
        numItems=50,
        avgItemsPerPatterns=10,
        avgPatternsPerSequence=5,
        seqSep=':',
        txnSep='\t',
        groupDist='poisson'
    )
    obj.create()
    obj.save('sequentialDatabase.csv')
    df = obj.getTransactions()
    print('Generated sequences:', df.shape[0])
    print('Runtime:', obj.getRuntime())
    print('Memory (RSS):', obj.getMemoryRSS())
    print('Memory (USS):', obj.getMemoryUSS())
