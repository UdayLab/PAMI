import random

def generate_random_dna_sequence(length):
    return ''.join(random.choice('ACGT') for _ in range(length))

def generate_random_dna_sequences(num_sequences, min_length, max_length):
    sequences = []
    for _ in range(num_sequences):
        sequence_length = random.randint(min_length, max_length)
        random_sequence = generate_random_dna_sequence(sequence_length)
        sequences.append(random_sequence)
    return sequences

# Set parameters
import pandas as pd
def generate(path,num_sequences,min_sequence_length,max_sequence_length):
# num_sequences = 1000
# min_sequence_length = 500
# max_sequence_length = 1000

    # Generate random DNA sequences
    random_sequences = generate_random_dna_sequences(num_sequences, min_sequence_length, max_sequence_length)

    # # Save sequences to a file or print them
    # id=1
    # l=[]
    # # with open('input.csv', 'w') as file:
    # for sequence in random_sequences:
    #     l.append([id,sequence])
    #     # file.write(str(id)+" "+sequence + '\n')
    #     id+=1
    df=pd.DataFrame(random_sequences)
    df.to_csv(path)
    
    
