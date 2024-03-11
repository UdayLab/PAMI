import random


class createTransactionalDatabase:
    """
    A class to generate a transactional database with random transactions.

    Attributes:
        databaseSize (int): The number of transactions in the database.
        numOfItems (int): The maximum number of items in each transaction.
        avgTransactionLength (int): The average length of transactions.
        transactions (list): List to store generated transactions.

    Methods:
        __init__(self, database_size, num_of_items, avg_transaction_length): Constructor to initialize the database parameters.
        generate(): Method to generate random transactions based on the specified parameters.
        generateRandomNumbers(n, target_sum): Method to generate a list of random numbers with a specified target sum.
        save(output_file, sep): Method to save the generated transactions to a CSV file.

    Usage:
        obj = TransactionalDatabase(database_size, num_of_items, avg_transaction_length)
        obj.generate()
        obj.save("transactional_output.csv")

    Credits:
        The complete program was written by A.Hemanth sree sai under the supervision of Professor Rage Uday Kiran.
    """

    def __init__(self, databaseSize: int, numOfItems: int, avgTransactionLength: int) -> None:
        """
        Constructor to initialize the database parameters.

        Parameters:
            databaseSize (int): The number of transactions in the database.
            numOfItems (int): The maximum number of items in each transaction.
            avgTransactionLength (int): The average length of transactions.
        """
        self.databaseSize = databaseSize
        self.numOfItems = numOfItems
        self.avgTransactionLength = avgTransactionLength
        self.transactions = list()

    def generate(self) -> None:
        """
        Method to generate random transactions based on the specified parameters.
        If avg_transaction_length exceeds num_of_items, an error message is printed.
        """
        if self.avgTransactionLength > self.numOfItems:
            print("Error: avg_transaction_length cannot exceed num_of_items.")
            return

        for _ in range(self.databaseSize):
            length = random.randint(1, self.avgTransactionLength * 2)
            transaction = [random.randint(1, self.numOfItems)
                           for _ in range(length)]
            self.transactions.append(transaction)

    def createRandomNumbers(self, n: int, targetsum: int) -> list[float]:
        """
        Method to generate a list of random numbers with a specified target sum.

        Parameters:
            n (int): Number of random numbers to generate.
            targetsum (int): Target sum for the generated random numbers.

        Returns:
            list: List of generated random numbers normalized and multiplied by the target sum.
        """
        rand_numbers = [random.uniform(0, 1) for _ in range(n)]
        rand_sum = sum(rand_numbers)
        normalizedNumbers = [num / rand_sum for num in rand_numbers]
        result = [round(num * targetsum) for num in normalizedNumbers]
        return result

    def save(self, outputfile: str, sep="\t") -> None:
        """
        Method to save the generated transactions to a CSV file.

        Parameters:
            outputfile (str): File path for the CSV file.
            sep (str): Separator for the CSV file (default is '\t').
        """
        with open(outputfile, 'w') as f:
            for transaction in self.transactions:
                # Adding the generated random numbers to each transaction
                randomNumbers = self.createRandomNumbers(13, 2000)
                transaction += randomNumbers

                f.write(f"{sep.join(map(str, transaction))}\n")


if __name__ == "__main__":
    obj = createTransactionalDatabase(10, 10, 5)
    obj.generate()

    # Saving transactions to a CSV file with generated random numbers
    obj.save("transactional_output-10.csv")
