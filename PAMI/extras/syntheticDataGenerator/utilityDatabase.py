import numpy as np
import pandas as pd
import random


class utilityDataGenerator:
    def __init__(self, databaseSize, numberOfItems, averageLengthOfTransaction,
                 minimumInternalUtilityValue, maximumInternalUtilityValue,
                 minimumExternalUtilityValue, maximumExternalUtilityValue):
        self.databaseSize = databaseSize
        self.numberOfItems = numberOfItems
        self.averageLengthOfTransaction = averageLengthOfTransaction
        self.minInternalUtilityValue = minimumInternalUtilityValue
        self.maxInternalUtilityValue = maximumInternalUtilityValue
        self.minExternalUtilityValue = minimumExternalUtilityValue
        self.maxExternalUtilityValue = maximumExternalUtilityValue
        self.entries = []
        self.ExternalUtilityData = self.generateExternalUtilityData()

    def generateExternalUtilityData(self):
        items = range(1, self.numberOfItems + 1)
        return {f'item{item}': random.randint(self.minExternalUtilityValue, self.maxExternalUtilityValue) for item in items}


    def generate(self):
        for entry_id in range(1, self.databaseSize + 1):
            entry_length = np.random.randint(1, self.averageLengthOfTransaction * 2)
            entry = np.random.randint(self.minInternalUtilityValue, self.maxInternalUtilityValue + 1,
                                      size=self.numberOfItems)
            entry_sum = entry.sum()
            self.entries.append((entry, entry_sum))

    def save(self, fileName):
        with open(fileName, 'w') as file:
            for idx, (entry, entry_sum) in enumerate(self.entries, start=1):
                entry_str = '\t'.join(map(str, entry))
                file.write(f'{idx}\t{entry_str}\t{entry_sum}\n')

    def saveItemsInternalUtilityValues(self, fileName):
        items = random.sample(range(1, self.numberOfItems + 1), self.numberOfItems)
        internal_utility_data = [np.random.randint(self.minInternalUtilityValue, self.maxInternalUtilityValue + 1) for _
                                 in items]
        data = {'Item': items, 'Internal Utility Value': internal_utility_data}
        df = pd.DataFrame(data)
        df.to_csv(fileName, sep='\t', index=False)

    def saveItemsExternalUtilityValues(self, fileName):
        items = random.sample(range(1, self.numberOfItems + 1), self.numberOfItems)
        data = {'Item': [f'item{item}' for item in items],
                'External Utility Value': list(self.ExternalUtilityData.values())}
        df = pd.DataFrame(data)
        df.to_csv(fileName, sep='\t', index=False)

    def getUtilityData(self):
        data = {'Entry ID': range(1, len(self.entries) + 1),
                'Entries': [entry for entry, _ in self.entries],
                'Sum': [entry_sum for _, entry_sum in self.entries]}
        df = pd.DataFrame(data)
        return df

    def getInternalUtilityData(self):
        items = random.sample(range(1, self.numberOfItems + 1), self.numberOfItems)
        InternalUtilityData = [np.random.randint(self.minInternalUtilityValue, self.maxInternalUtilityValue + 1) for _
                                 in items]
        data = {'Item': items, 'Internal Utility Value': InternalUtilityData}
        df = pd.DataFrame(data)
        return df

    def getExternalUtilityData(self):
        items = random.sample(range(1, self.numberOfItems + 1), self.numberOfItems)
        data = {'Item': [f'item{item}' for item in items],
                'External Utility Value': list(self.ExternalUtilityData.values())}
        df = pd.DataFrame(data)
        return df

    def generateAndPrintItemPairs(self):
        items = random.sample(range(1, self.numberOfItems + 1), 2)
        item1_id = f'item{items[0]}'
        item2_id = f'item{items[1]}'
        item1_value = self.ExternalUtilityData[item1_id]
        item2_value = self.ExternalUtilityData[item2_id]
        sum_values = item1_value + item2_value
        print(f"{item1_id} value: {item1_value}\t{item2_id} value: {item2_value}\tSum of values: {sum_values}")

        # Separate the sum with ' : '
        print(f"{item1_value}:{item2_value}:{sum_values}")


if __name__ == "__main__":
    data_generator = utilityDataGenerator(100000, 2000, 10, 1, 100, 1, 10)
    data_generator.generate()
    data_generator.save("utility_data-6.csv")
    data_generator.saveItemsInternalUtilityValues("items_internal_utility.csv")
    data_generator.saveItemsExternalUtilityValues("items_external_utility.csv")
    utility_data = data_generator.getUtilityData()
    InternalUtilityData = data_generator.getInternalUtilityData()
    ExternalUtilityData = data_generator.getExternalUtilityData()

    for _ in range(10):  # Print pairs for demonstration, adjust the range as needed
        data_generator.generateAndPrintItemPairs()
