import random
import string
import time
import os
import psutil


class symbolicSequenceDatabase:
    def __init__(self, sequenceSize=100000, numberOfSymbols=10):
        self.sequenceSize = sequenceSize
        self.numberOfSymbols = numberOfSymbols
        self.process = psutil.Process(os.getpid())

        # Full allowed ASCII pool: digits + letters + top-row specials
        self.symbolPool = list(string.digits + string.ascii_letters + "!@#$%^&*()")
        self.maxSymbols = len(self.symbolPool)

        if not (1 <= self.numberOfSymbols <= self.maxSymbols):
            raise ValueError(f"numberOfSymbols must be between 1 and {self.maxSymbols}")

    def create(self):
        self.startTime = time.time()

        # Choose subset of symbols to use
        self.symbols = random.sample(self.symbolPool, self.numberOfSymbols)

        # Generate one long sequence using chosen symbols
        self.sequence = ''.join(random.choices(self.symbols, k=self.sequenceSize))

        self.endTime = time.time()

        self.memoryRSS = self.process.memory_info().rss
        self.memoryUSS = self.process.memory_full_info().uss

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(self.sequence)

    def getSequence(self):
        return self.sequence

    def getRuntime(self):
        return self.endTime - self.startTime

    def getMemoryRSS(self):
        return self.memoryRSS

    def getMemoryUSS(self):
        return self.memoryUSS
