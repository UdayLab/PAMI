
class associationRuleMiner:
    """
    temporalDatabaseStats is class to get stats of database.

        Attributes:
        ----------
        frequentPattern : list or dict
            list
        measure: condition to calculate the strength of rule
            str
        threshold: condition to satisfy
            int

        Methods:
        -------
        startMine()

    """

    def __init__(self, frequentPatterns, measure, threshold):
        """
        :param inputFile: input file name or path
        :type inputFile: str
        :param sep:
        """
        self.frequentPattern = frequentPatterns
        self.measure = measure
        self.threshold = threshold

    def save(self, prefix, suffix):
        print(prefix, suffix)

    def generation(self, prefix, suffix):
        if len(suffix) == 0:
            return
        for i in range(len(suffix)):
            prefix = prefix + suffix[i]
            sample = str()
            for j in suffix[i+1:]:
                sample = sample + j
                print(prefix, sample)
                self.generation(prefix, sample)

    def associationRuleMiner(self):
        flist = []
        for i in self.frequentPattern.keys():
            for j in i:
                if j not in flist:
                    flist.append(j)
        for i in range(len(flist)):
            sample = str()
            for j in flist[i+1:]:
                sample = sample + j
                print(flist[i], sample)
                self.generation(flist[i], sample)



if __name__ == '__main__':
    data = {'c':2, 'cb':2, 'b':5,
            'bd':3, 'ba':3, 'd':5, 'da':3, 'a':5}
    ap = associationRuleMiner(data, "confidence", 50)
    ap.associationRuleMiner()