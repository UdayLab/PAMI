import random
import time
import psutil
import os

class NucleotideSequenceGenerator:
    """
    :Description: NucleotideSequenceGenerator generates a random DNA or RNA sequence with specified GC content and length.
    :Attributes:
        sequence_length: int
            Length of the generated sequence.
        gc_content: float
            Desired GC content as a decimal (e.g., 0.5 for 50%).
        is_rna: bool
            True for RNA sequence, False for DNA sequence.
        sequence: str
            Generated DNA or RNA sequence.
        memoryUSS : float
            Stores the total amount of USS memory consumed by the program.
        memoryRSS : float
            Stores the total amount of RSS memory consumed by the program.
        startTime : float
            Records the start time of the sequence generation process.
        endTime : float
            Records the completion time of the sequence generation process.
    :Methods:
        create:
            Generates the random DNA or RNA sequence.
        save:
            Saves the generated sequence to a user-specified file.
        getSequence:
            Returns the generated sequence.
        getMemoryUSS:
            Retrieves the total amount of USS memory consumed by the process.
        getMemoryRSS:
            Retrieves the total amount of RSS memory consumed by the process.
        getRuntime:
            Retrieves the total runtime taken by the sequence generation process.
    """

    def __init__(self, sequence_length, gc_content, is_rna=False):
        self.sequence_length = sequence_length
        self.gc_content = gc_content
        self.is_rna = is_rna
        self.sequence = ""
        self._startTime = float()
        self._endTime = float()
        self._memoryUSS = float()
        self._memoryRSS = float()

    def create(self):
        self._startTime = time.time()
        
        # Define nucleotide choices based on DNA or RNA
        nucleotides = "GCAU" if self.is_rna else "GCAT"

        # Calculate number of G/C and A/T (or A/U) bases
        gc_count = int(self.sequence_length * self.gc_content)
        at_count = self.sequence_length - gc_count

        # Build the sequence with the correct proportion of GC and AT
        sequence_list = (
            random.choices("GC", k=gc_count) +
            random.choices("AU" if self.is_rna else "AT", k=at_count)
        )

        # Shuffle to randomize order
        random.shuffle(sequence_list)

        # Convert list to string
        self.sequence = "".join(sequence_list)

        self._endTime = time.time()

    def save(self, filename):
        with open(filename, 'w') as file:
            file.write(self.sequence)

    def getSequence(self):
        return self.sequence

    def getMemoryUSS(self) -> float:
        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        return self._memoryUSS

    def getMemoryRSS(self) -> float:
        process = psutil.Process(os.getpid())
        self._memoryRSS = process.memory_info().rss
        return self._memoryRSS

    def getRuntime(self) -> float:
        return self._endTime - self._startTime

