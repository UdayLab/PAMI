import random
import warnings

warnings.filterwarnings("ignore")

def generate_transactional_dataset(num_transactions, items, max_items_per_transaction):
    dataset = []
    for _ in range(num_transactions):
        num_items = random.randint(1, max_items_per_transaction)
        transaction = random.sample(items, num_items)
        dataset.append(transaction)
    return dataset

# Example usage:
# num_distinct_items=20
# num_transactions = 1000
# max_items_per_transaction = 20
# items=["item-{}".format(i) for i in range(1,num_distinct_items+1)]
# dataset = generate_transactional_dataset(num_transactions, items, max_items_per_transaction)
# print(dataset)
