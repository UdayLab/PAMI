import numpy as np
import pandas as pd
import random

class UtilityDataGenerator:
    def __init__(self, numberOfItems):
        self.numberOfItems = numberOfItems
        self.alphabets = [chr(random.randint(65, 90)) for _ in range(self.numberOfItems)]
        self.internal_utility_values = [random.randint(1, 100) for _ in range(self.numberOfItems)]
        self.external_utility_values = [random.randint(1, 10) for _ in range(self.numberOfItems)]
        self.pairs = [(a, b, a + b) for a, b in zip(self.internal_utility_values, self.external_utility_values)]

    def generate_pairs(self):
        data = {'Item ID': range(1, self.numberOfItems + 1),
                'Internal Utility Value': self.internal_utility_values,
                'External Utility Value': self.external_utility_values,
                'Sum of Utilities': [a + b for a, b in zip(self.internal_utility_values, self.external_utility_values)]}
        df = pd.DataFrame(data)

        return df

    def save(self, file_name):
        pairs_df = self.generate_pairs()
        pairs_df.to_csv(file_name, sep='\t', index=False)

    def print_pairs(self):
        pairs_df = self.generate_pairs()
        print(pairs_df)

if __name__ == "__main__":
    data_generator = UtilityDataGenerator(2000)
    data_generator.save("utility_data.csv")
    data_generator.print_pairs()
