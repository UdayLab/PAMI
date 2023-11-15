import numpy as np
import pandas as pd

class TemporalDataGenerator:
    def __init__(self, transaction_size, num_of_items, avg_transaction_length):
        self.transaction_size = transaction_size
        self.num_of_items = num_of_items
        self.avg_transaction_length = avg_transaction_length
        self.transactions = []

    def GenerateData(self):
        for transaction_id in range(self.transaction_size):
            transaction_length = np.random.randint(1, self.avg_transaction_length * 2)
            transaction = np.random.choice(range(1, self.num_of_items + 1), size=transaction_length)
            self.transactions.append(transaction)

    def Save(self, output_file):
        with open(output_file, 'w') as file:
            for idx, transaction in enumerate(self.transactions, start=1):
                transaction_str = '\t'.join(map(str, transaction))
                file.write(f'{idx}\t{transaction_str}\n')

if __name__ == "__main__":
    data_generator = TemporalDataGenerator(10, 10, 5)
    data_generator.GenerateData()
    data_generator.Save("temporal_data.csv")
