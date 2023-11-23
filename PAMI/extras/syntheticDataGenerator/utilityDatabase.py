import numpy as np
import pandas as pd
import random


class UtilityDataGenerator:
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
        self.ExternalUtilityData = self.GenerateExternalUtilityData()

    def GenerateExternalUtilityData(self):
        items = range(1, self.numberOfItems + 1)
        ExternalUtilityData = {f'item{item}': random.randint(100, 900) for item in items}
        return ExternalUtilityData

    def Generate(self):
        for entry_id in range(1, self.databaseSize + 1):
            entry_length = np.random.randint(1, self.averageLengthOfTransaction * 2)
            entry = np.random.randint(self.minInternalUtilityValue, self.maxInternalUtilityValue + 1,
                                      size=self.numberOfItems)
            entry_sum = entry.sum()
            self.entries.append((entry, entry_sum))

    def Save(self, fileName):
        with open(fileName, 'w') as file:
            for idx, (entry, entry_sum) in enumerate(self.entries, start=1):
                entry_str = '\t'.join(map(str, entry))
                file.write(f'{idx}\t{entry_str}\t{entry_sum}\n')

    def SaveItemsInternalUtilityValues(self, fileName):
        items = random.sample(range(1, self.numberOfItems + 1), self.numberOfItems)
        internal_utility_data = [np.random.randint(self.minInternalUtilityValue, self.maxInternalUtilityValue + 1) for _
                                 in items]
        data = {'Item': items, 'Internal Utility Value': internal_utility_data}
        df = pd.DataFrame(data)
        df.to_csv(fileName, sep='\t', index=False)

    def Saveitemsexternalutilityvalues(self, fileName):
        items = random.sample(range(1, self.numberOfItems + 1), self.numberOfItems)
        data = {'Item': [f'item{item}' for item in items],
                'External Utility Value': list(self.ExternalUtilityData.values())}
        df = pd.DataFrame(data)
        df.to_csv(fileName, sep='\t', index=False)

    def GetUtilityData(self):
        data = {'Entry ID': range(1, len(self.entries) + 1),
                'Entries': [entry for entry, _ in self.entries],
                'Sum': [entry_sum for _, entry_sum in self.entries]}
        df = pd.DataFrame(data)
        return df

    def GetInternalUtilityData(self):
        items = random.sample(range(1, self.numberOfItems + 1), self.numberOfItems)
        InternalUtilityData = [np.random.randint(self.minInternalUtilityValue, self.maxInternalUtilityValue + 1) for _
                                 in items]
        data = {'Item': items, 'Internal Utility Value': InternalUtilityData}
        df = pd.DataFrame(data)
        return df

    def GetExternalUtilityData(self):
        items = random.sample(range(1, self.numberOfItems + 1), self.numberOfItems)
        data = {'Item': [f'item{item}' for item in items],
                'External Utility Value': list(self.ExternalUtilityData.values())}
        df = pd.DataFrame(data)
        return df

    def GenerateAndPrintItemPairs(self):
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
    data_generator = UtilityDataGenerator(100000, 2000, 10, 1, 100, 1, 10)
    data_generator.Generate()
    data_generator.Save("utility_data-6.csv")
    data_generator.SaveItemsInternalUtilityValues("items_internal_utility.csv")
    data_generator.Saveitemsexternalutilityvalues("items_external_utility.csv")
    utility_data = data_generator.GetUtilityData()
    InternalUtilityData = data_generator.GetInternalUtilityData()
    ExternalUtilityData = data_generator.GetExternalUtilityData()

    for _ in range(10):  # Print pairs for demonstration, adjust the range as needed
        data_generator.GenerateAndPrintItemPairs()



















































