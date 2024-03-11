import random as _rd


class createTemporalDatabase:
    """
    This class creates a synthetic temporal database.

    Attributes:
        totalTransactions (int): Number of transactions.
        numOfItems (int): Number of items.
        avgTransactionLength (int): The length of average transaction.

    Methods:
        __init__(totalTransactions, numOfItems, avgTransactionLength)
            Constructor to initialize the database parameters.
        save(outputFile)
            Save the generated temporal database to a CSV file.

    Credits:
        The complete program was written by P.Likhitha under the supervision of Professor Rage Uday Kiran.
    """

    def __init__(self, totalTransactions: int, numOfItems: int, avgTransactionLength: int) -> None:
        """
        Constructor to initialize the database parameters.

        Parameters:
            totalTransactions (int): Number of transactions.
            numOfItems (int): Number of items.
            avgTransactionLength (int): The length of average transaction.
        """
        self.totalTransactions = totalTransactions
        self.numOfItems = numOfItems
        self.avgTransactionLength = avgTransactionLength
        self.transactions = []

    def save(self, outputFile: str) -> None:
        """
        Save the generated temporal database to a CSV file.

        Parameters:
            outputFile (str): File name or path to store the CSV file.
        """
        if self.avgTransactionLength > self.numOfItems:
            print("Error: avgTransactionLength cannot exceed numOfItems.")
            return

        count = 1
        with open(outputFile, 'w') as writer:
            for _ in range(self.totalTransactions):
                length = _rd.randint(1, self.avgTransactionLength + 20)
                st = str(count) + '\t'
                randomNumbers = self.createRandomNumbers(13, 2000)

                # Checking if avgTransactionLength exceeds numOfItems
                if self.avgTransactionLength > self.numOfItems:
                    print("Error: avgTransactionLength cannot exceed numOfItems.")
                    return

                for _ in range(length):
                    item = _rd.randint(1, self.numOfItems)
                    st = st + str(item) + '\t'
                writer.write("%s \n" % st)
                count += 1

    def createRandomNumbers(self, n: int, targetSum: int) -> list[float]:
        """
        Generate a list of random numbers with a specified target sum.

        Parameters:
            n (int): Number of random numbers to generate.
            targetSum (int): Target sum for the generated random numbers.

        Returns:
            list: List of generated random numbers normalized and multiplied by the target sum.
        """
        randNumbers = [_rd.uniform(0, 1) for _ in range(n)]
        randSum = sum(randNumbers)
        normalizedNumbers = [num / randSum for num in randNumbers]
        result = [round(num * targetSum) for num in normalizedNumbers]
        return result


if __name__ == "__main__":
    ap = createTemporalDatabase(100000, 870, 10)
    ap.save("temporal_T10-16.csv")
else:
    print("Error! The number of input parameters does not match the total number of parameters provided")
