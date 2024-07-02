import random

def generate_transactional_dataset(num_transactions, items, max_items_per_transaction):
    dataset = []
    for _ in range(num_transactions):
        num_items = random.randint(1, max_items_per_transaction)
        transaction = random.sample(items, num_items)
        dataset.append(transaction)
    return dataset
