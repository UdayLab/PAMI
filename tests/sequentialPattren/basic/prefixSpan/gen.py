import random
import warnings
warnings.filterwarnings("ignore")

def generate_sequentional_dataset(num_transactions, items, max_items_per_sequence,max_sequence_per_transaction,sep="-1"):
    dataset = []
    for _ in range(num_transactions):

        num_seq=random.randint(1, max_sequence_per_transaction)
        itemset=[]
        for i in range(num_seq):
            num_items = random.randint(1, max_items_per_sequence)
            for item in  random.sample(items, num_items):
                itemset.append(item)
            itemset.append(sep)
        dataset.append(itemset)
    return dataset


# num_distinct_items=20
# num_transactions = 1000
# max_items_per_transaction = 20
# items=["item-{}".format(i) for i in range(1,num_distinct_items+1)]

# dataset = generate_transactional_dataset(num_transactions, items, max_items_per_transaction)
# print(dataset)