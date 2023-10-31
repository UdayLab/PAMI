import random


class generateTransactional:
    __transactionSize: int
    __numOfItems: int
    __avgTransactionLength: int
    __transactions: list[list[int]]

    def __init__(self, transactionSize: int, numOfItems: int, avgTransactionLength: int) -> None:
        self.__transactionSize = transactionSize
        self.__numOfItems = numOfItems
        self.__avgTransactionLength = avgTransactionLength

        self.__transactions = list()

    def generate(self) -> None:
        for tid in range(self.__transactionSize):
            length = random.randint(1, self.__avgTransactionLength * 2)
            transaction = [random.randint(1, self.__numOfItems)
                           for _ in range(length)]
            self.__transactions.append(transaction)

    def save(self, outputFile: str, sep="\t") -> None:
        with open(outputFile, 'w') as f:
            for transaction in self.__transactions:
                f.write(f"{sep.join(map(str, transaction))}\n")


if __name__ == "__main__":
    obj = generateTransactional(10, 10, 5)
    obj.generate()
    obj.save("transactional_test.csv")
