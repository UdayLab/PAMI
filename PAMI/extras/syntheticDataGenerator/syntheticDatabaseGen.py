import random as _rd


class CreateSyntheticUtility:
    """
    This class creates a synthetic utility database.

    Attributes:
        total_transactions (int): Number of transactions.
        num_of_items (int): Number of items.
        max_util_range (int): Maximum utility range.
        avg_transaction_length (int): The length of average transaction.

    Methods:
        create_utility_database(output_file)
            Create utility database and store it in the specified output file.

    Credits:
        The complete program was written by A.Hemanth sree sai under the supervision of Professor Rage Uday Kiran.
    """

    def __init__(self, total_transactions: int, num_of_items: int, max_util_range: int, avg_transaction_length: int) -> None:
        """
        Constructor to initialize the database parameters.

        Parameters:
            total_transactions (int): Number of transactions.
            num_of_items (int): Number of items.
            max_util_range (int): Maximum utility range.
            avg_transaction_length (int): The length of average transaction.
        """
        self.total_transactions = total_transactions
        self.num_of_items = num_of_items
        self.max_util_range = max_util_range
        self.avg_transaction_length = avg_transaction_length

    def create_utility_database(self, output_file: str) -> None:
        """
        Create utility database and store it in the specified output file.

        Parameters:
            output_file (str): File name or path to store the database.
        """
        if self.avg_transaction_length > self.num_of_items:
            print("Error: avg_transaction_length cannot exceed num_of_items.")
            return

        with open(output_file, 'w') as writer:
            for _ in range(self.total_transactions):
                length = _rd.randint(1, self.avg_transaction_length + 20)
                items = [_rd.randint(1, self.num_of_items) for _ in range(length)]
                utilities = [_rd.randint(1, self.max_util_range) for _ in range(length)]

                # Generating 13 random numbers with a target sum of 2000
                random_numbers = self.generate_random_numbers(13, 2000)

                # Checking if avgTransactionLength exceeds numOfItems
                if self.avg_transaction_length > self.num_of_items:
                    print("Error: avg_transaction_length cannot exceed num_of_items.")
                    return

                st = '\t'.join(map(str, items)) + '\t:' + str(sum(utilities)) + ':'
                st1 = '\t'.join(map(str, random_numbers)) + '\t'

                writer.write(f"{st}{st1}\n")

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
    ap = CreateSyntheticUtility(100000, 870, 100, 10)
    ap.create_utility_database("T10_util.csv")
else:
    print("Error! The number of input parameters does not match the total number of parameters provided")
