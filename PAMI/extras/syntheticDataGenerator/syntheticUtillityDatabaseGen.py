import random as _rd


class CreateSyntheticUtility:
    """
    This class creates a synthetic utility database.

    Attributes:
        totalTransactions (int): Number of transactions.
        numOfItems (int): Number of items.
        maxUtilRange (int): Maximum utility range.
        avgTransactionLength (int): The length of average transaction.

    Methods:
        createUtilityDatabase(outputFile)
            Create utility database and store it in the specified output file.

    Credits:
        The complete program was written by A.Hemanth sree sai under the supervision of Professor Rage Uday Kiran.
    """

    def __init__(self, totalTransactions: int, numOfItems: int, maxUtilRange: int, avgTransactionLength: int) -> None:
        """
        Constructor to initialize the database parameters.

        Parameters:
            totalTransactions (int): Number of transactions.
            numOfItems (int): Number of items.
            maxUtilRange (int): Maximum utility range.
            avgTransactionLength (int): The length of average transaction.
        """
        self.totalTransactions = totalTransactions
        self.numOfItems = numOfItems
        self.maxUtilRange = maxUtilRange
        self.avgTransactionLength = avgTransactionLength

    def createUtilityDatabase(self, outputFile: str) -> None:
        """
        Create utility database and store it in the specified output file.

        Parameters:
            outputFile (str): File name or path to store the database.
        """
        if self.avgTransactionLength > self.numOfItems:
            print("Error: avgTransactionLength cannot exceed numOfItems.")
            return

        with open(outputFile, 'w') as writer:
            for _ in range(self.totalTransactions):
                length = _rd.randint(1, self.avgTransactionLength + 20)
                items = [_rd.randint(1, self.numOfItems) for _ in range(length)]
                utilities = [_rd.randint(1, self.maxUtilRange) for _ in range(length)]

                # Generating 13 random numbers with a target sum of 2000
                randomNumbers = self.generateRandomNumbers(13, 2000)

                # Checking if avgTransactionLength exceeds numOfItems
                if self.avgTransactionLength > self.numOfItems:
                    print("Error: avgTransactionLength cannot exceed numOfItems.")
                    return

                st = '\t'.join(map(str, items)) + '\t:' + str(sum(utilities)) + ':'
                st1 = '\t'.join(map(str, randomNumbers)) + '\t'

                writer.write(f"{st}{st1}\n")

    def generateRandomNumbers(self, n: int, targetSum: int) -> list[float]:
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
    ap = CreateSyntheticUtility(100000, 870, 100, 10)
    ap.createUtilityDatabase("T10_util.csv")
else:
    print("Error! The number of input parameters does not match the total number of parameters provided")
