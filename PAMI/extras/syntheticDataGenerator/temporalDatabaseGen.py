import random as _rd


class CreateSyntheticTemporal:
    """
    This class creates a synthetic temporal database.

    Attributes:
        total_transactions (int): Number of transactions.
        num_of_items (int): Number of items.
        avg_transaction_length (int): The length of average transaction.

    Methods:
        create_temporal_database(output_file)
            Create temporal database and store it in the specified output file.

    Credits:
        The complete program was written by A.Hemanth sree sai under the supervision of Professor Rage Uday Kiran.
    """

    def __init__(self, total_transactions: int, num_of_items: int, avg_transaction_length: int) -> None:
        """
        Constructor to initialize the database parameters.

        Parameters:
            total_transactions (int): Number of transactions.
            num_of_items (int): Number of items.
            avg_transaction_length (int): The length of average transaction.
        """
        self.total_transactions = total_transactions
        self.num_of_items = num_of_items
        self.avg_transaction_length = avg_transaction_length

    def create_temporal_database(self, output_file: str) -> None:
        """
        Create temporal database and store it in the specified output file.

        Parameters:
            output_file (str): File name or path to store the database.
        """
        if self.avg_transaction_length > self.num_of_items:
            print("Error: avg_transaction_length cannot exceed num_of_items.")
            return

        count = 1
        with open(output_file, 'w') as writer:
            for _ in range(self.total_transactions):
                length = _rd.randint(1, self.avg_transaction_length + 20)
                st = str(count) + '\t'
                random_numbers = self.generate_random_numbers(13, 2000)

                # Checking if avgTransactionLength exceeds numOfItems
                if self.avg_transaction_length > self.num_of_items:
                    print("Error: avg_transaction_length cannot exceed num_of_items.")
                    return

                for _ in range(length):
                    item = _rd.randint(1, self.num_of_items)
                    st = st + str(item) + '\t'
                writer.write("%s \n" % st)
                count += 1

    def generate_random_numbers(self, n: int, target_sum: int) -> list[float]:
        """
        Generate a list of random numbers with a specified target sum.

        Parameters:
            n (int): Number of random numbers to generate.
            target_sum (int): Target sum for the generated random numbers.

        Returns:
            list: List of generated random numbers normalized and multiplied by the target sum.
        """
        rand_numbers = [_rd.uniform(0, 1) for _ in range(n)]
        rand_sum = sum(rand_numbers)
        normalized_numbers = [num / rand_sum for num in rand_numbers]
        result = [round(num * target_sum) for num in normalized_numbers]
        return result


if __name__ == "__main__":
    ap = CreateSyntheticTemporal(100000, 870, 10)
    ap.create_temporal_database("temporal_T10.csv")
else:
    print("Error! The number of input parameters does not match the total number of parameters provided")
