import numpy as np
import pandas as pd
import sys, psutil, os, time


class transactionalDatabase:

    def __init__(self, dataBaseSize, avgItemsPerTransaction, itemsNo, sep='\t') -> None:

        self.dataBaseSize = dataBaseSize
        self.avgItemsPerTransaction = avgItemsPerTransaction
        self.itemsNo = itemsNo
        self.sep = sep
        self.data = []
        self._startTime = float()
        self._endTime = float()
        self._memoryUSS = float()
        self._memoryRSS = float()


    def coin_flip(self,probability):
        return np.random.choice([0, 1], p=[1 - probability, probability]) == 1



    def noOfItemsPerTransaction(self,dataBaseSize,averageItemsPerTransaction,itemsNo):

        #generating random numbers with size of dataBaseSize
        transactionSize = np.random.rand(dataBaseSize)

        #sum of the values in the transactionSize array - sumTransactions
        sumTransactions = np.sum(transactionSize)

        #weights of the values in the array - transactionSize
        weights = transactionSize/sumTransactions

        #sumResultant -> whose average is equal to averageItemsPerTransaction's value
        sumResultant = averageItemsPerTransaction*dataBaseSize

        #Getting new values whose average is averageItemsPerTransaction
        newValues = np.round(weights*sumResultant)

        #changing the values in numpy array to int
        valuesInt = newValues.astype(int)

        #finding if there are any 0's in the array
        indexZero = np.where(valuesInt==0)[0]

        #Adding +1 to the transactions which have 0
        if len(indexZero)==0:
            return valuesInt
        else:
            for i in indexZero:
                valuesInt[i]+=1

        return valuesInt

    def create(self):
        self._startTime = time.time()
        noofItemsperTrans = self.noOfItemsPerTransaction(self.dataBaseSize, self.avgItemsPerTransaction, self.itemsNo)
        for i in range(self.dataBaseSize):
            self.data.append(np.random.choice(range(1,self.itemsNo+1), noofItemsperTrans[i], replace=False))


    def save(self, filename):
        with open(filename, 'w') as f:
            for line in self.data:
                f.write(str(self.sep).join(map(str, line)) + '\n')


    def getdataasDataframe(self,sep='\t'):
        column = 'Transaction'
        dataFrame = pd.DataFrame(colums=column)
        dataFrame[column] = [sep.join(map(str,line) for line in self.data)]
        return dataFrame


    def getMemoryUSS(self) -> float:

        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        return self._memoryUSS

    def getMemoryRSS(self) -> float:

        process = psutil.Process(os.getpid())
        self._memoryRSS = process.memory_info().rss
        return self._memoryRSS

    def getRuntime(self) -> float:
        self._endTime = time.time()
        return self._endTime - self._startTime

if __name__ == "__main__":

    if len(sys.argv) == 5:
        obj = transactionalDatabase(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        obj.create()
        obj.save(sys.argv[4])
        print("Total Memory in USS:", obj.getMemoryUSS())
        print("Total Memory in RSS", obj.getMemoryRSS())
        print("Total ExecutionTime in ms:", obj.getRuntime())
    elif len(sys.argv) == 6:
        obj = transactionalDatabase(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        obj.create()
        obj.save(sys.argv[5])
        print("Total Memory in USS:", obj.getMemoryUSS())
        print("Total Memory in RSS", obj.getMemoryRSS())
        print("Total ExecutionTime in ms:", obj.getRuntime())
    else:
        raise ValueError(
            "Invalid number of arguments. Args: <databaseSize> <avgItemsPerTransaction> <noOfItems> <filename> or Args: <databaseSize> <avgItemsPerTransaction> <noOfItems> <sep> <filename>")

