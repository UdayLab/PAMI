import random


class generateUncertainTemporal:
    __transactionSize: int
    __numOfItems: int
    __avgTransactionLength: int
    __significant: int
    __transactions: list[list[int]]
    __probabilitis: list[list[float]]

    def __init__(self, transactionSize: int, numOfItems: int, avgTransactionLength: int, significant=2) -> None:
        self.__transactionSize = transactionSize
        self.__numOfItems = numOfItems
        self.__avgTransactionLength = avgTransactionLength
        self.__significant = significant

        self.__transactions = list()
        self.__probabilitis = list()

    def generate(self) -> None:
        for tid in range(self.__transactionSize):
            length = random.randint(1, self.__avgTransactionLength * 2)
            transaction = [random.randint(1, self.__numOfItems)
                           for _ in range(length)]
            self.__transactions.append(transaction)
            probability = [round(random.uniform(0, 1), self.__significant)
                           for _ in range(length)]
            self.__probabilitis.append(probability)

    def save(self, outputFile: str, sep="\t") -> None:
        with open(outputFile, 'w') as f:
            for tid, (transaction, probability) in enumerate(zip(self.__transactions, self.__probabilitis)):
                f.write(
                    f"{tid+1}{sep}{sep.join(map(str, transaction))}:{round(sum(probability), self.__significant)}:{sep.join(map(str, probability))}\n")


if __name__ == "__main__":
    obj = generateUncertainTemporal(10, 10, 5)
    obj.generate()
    obj.save("uncertainTemporal_test.csv")
