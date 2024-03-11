import random


class TransactionalDatabase:
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
    """

    def __init__(self, database_size: int, num_of_items: int, avg_transaction_length: int) -> None:
        """
        Constructor to initialize the database parameters.

        Parameters:
            database_size (int): The number of transactions in the database.
            num_of_items (int): The maximum number of items in each transaction.
            avg_transaction_length (int): The average length of transactions.
        """
        self.databaseSize = database_size
        self.numOfItems = num_of_items
        self.avgTransactionLength = avg_transaction_length
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

    def generateRandomNumbers(self, n: int, target_sum: int) -> list[float]:
        """
        Method to generate a list of random numbers with a specified target sum.

        Parameters:
            n (int): Number of random numbers to generate.
            target_sum (int): Target sum for the generated random numbers.

        Returns:
            list: List of generated random numbers normalized and multiplied by the target sum.
        """
        rand_numbers = [random.uniform(0, 1) for _ in range(n)]
        rand_sum = sum(rand_numbers)
        normalized_numbers = [num / rand_sum for num in rand_numbers]
        result = [round(num * target_sum) for num in normalized_numbers]
        return result

    def save(self, output_file: str, sep="\t") -> None:
        """
        Method to save the generated transactions to a CSV file.

        Parameters:
            output_file (str): File path for the CSV file.
            sep (str): Separator for the CSV file (default is '\t').
        """
        with open(output_file, 'w') as f:
            for transaction in self.transactions:
                # Adding the generated random numbers to each transaction
                random_numbers = self.generateRandomNumbers(13, 2000)
                transaction += random_numbers

                f.write(f"{sep.join(map(str, transaction))}\n")


if __name__ == "__main__":
    obj = TransactionalDatabase(10, 10, 5)
    obj.generate()

    # Saving transactions to a CSV file with generated random numbers
    obj.save("transactional_output.csv")
