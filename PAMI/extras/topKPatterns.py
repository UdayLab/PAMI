class topKPatterns:
    """
    find top k length patterns in input file.

    Attributes:
    -----------
        inputFile : str
            input file name or path
        k : int
            rank of pattern length. default is 10
        sep : str
            separator which separate patterns in input file. default is tab space

    Methods:
    -------
        getTopKPatterns()
            return top k patterns as dict
        storeTopKPatterns(outputFile)
            store top k patterns into output file.

    """
    def __init__(self, inputFile, k=10, sep='\t'):
        self.inputFile = inputFile
        self.k = k
        self.sep = sep

    def getTopKPatterns(self):
        """
        get top k length patterns. user can defined k value.
        :return: top k length patterns as dictionary. top k patterns = {patternId: pattern}
        """
        with open(self.inputFile, 'r') as f:
            patterns = [[item for item in line.strip().split(':')][0].split(self.sep)[:-1] for line in f]
        patterns = sorted(patterns, key=lambda x: len(x[0]), reverse=True)
        return {patternId: patterns[patternId - 1] for patternId in range(1, int(self.k)+1)}

    def save(self, outputFile):
        """
        store top k length patterns into file. user can defined k value.
        :param outputFile: output file name or path
        :type outputFile: str
        """
        with open(self.inputFile, 'r') as f:
            patterns = [[item for item in line.strip().split(':')][0].split(self.sep)[:-1] for line in f]
            patterns = sorted(patterns, key=lambda x: len(x[0]), reverse=True)
        with open(outputFile, 'w') as f:
            patternId = 1
            for pattern in patterns[:self.k]:
                for item in pattern:
                    f.write(f'{patternId}\t{item}\n')
