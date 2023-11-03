import numpy as np
import pandas as pd

class TemporalDataGenerator:
    def __init__(self, transaction_size, num_of_items, avg_transaction_length):
        self.transaction_size = transaction_size
        self.num_of_items = num_of_items
        self.avg_transaction_length = avg_transaction_length
        self.transactions = []

    def generate_data(self):
        for transaction_id in range(self.transaction_size):
            transaction_length = np.random.randint(1, self.avg_transaction_length * 2)
            transaction = np.random.choice(range(1, self.num_of_items + 1), size=transaction_length)
            self.transactions.append(transaction)

    def save_to_csv(self, output_file):
        data = {'Transaction ID': range(1, len(self.transactions) + 1), 'Items': self.transactions}
        df = pd.DataFrame(data)
        df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    data_generator = TemporalDataGenerator(10, 10, 5)
    data_generator.generate_data()
    data_generator.save_to_csv("temporal_test.csv")
