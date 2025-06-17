import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt



class SymbolicSequentialDatabase:
    def __init__(self, filename):
        self.filename = filename
        self.sequence = ""

    def run(self):
        # Load file and concatenate all sequences into one string
        with open(self.filename, 'r') as f:
            self.sequence = ''.join(f.read().splitlines())


        self.length = len(self.sequence)
        self.symbolCounts = Counter(self.sequence)
        self.numUniqueSymbols = len(self.symbolCounts)

    def printStats(self):
        print(f"Total Number of Symbols: {self.numUniqueSymbols}")
        print(f"Total Size of Sequence:  {self.length}")
        print("Symbol Frequencies:")
        for symbol, count in self.symbolCounts.items():
            print(f"  '{symbol}': {count}")

    def plotGraphs(self):
        if not hasattr(self, 'symbolCounts'):
            raise RuntimeError("You must call run() before plotGraphs().")

        symbols, counts = zip(*sorted(self.symbolCounts.items(), key=lambda x: x[1], reverse=True))

        plt.figure(figsize=(12, 6))
        plt.bar(symbols, counts, width=0.6)
        plt.title("Symbol Frequency Distribution")
        plt.xlabel("Symbol")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
