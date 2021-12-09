
class RuleMiner:
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
        self.finalPatterns = {}

    def save(self, prefix, suffix):
        print(prefix, suffix)

    def generation(self, pattern):
        print("Hello")

    def RuleMiner(self):
        for x, y in self.frequentPattern.items():
            if len(x) == 2:
                conf = (self.frequentPattern[x[0]] + self.frequentPattern[x[1]]) / self.frequentPattern[x[0]]
                conf1 = (self.frequentPattern[x[0]] + self.frequentPattern[x[1]]) / self.frequentPattern[x[1]]
                if conf >= self.threshold:
                    if x[0] in self.finalPatterns:
                        self.finalPatterns[x[0]].update({x[1]:conf})
                    if x[0] not in self.finalPatterns:
                        self.finalPatterns[x[0]] = {x[1]:conf}
                if conf1 >= self.threshold:
                    if x[1] in self.finalPatterns:
                        self.finalPatterns[x[1]].update({str(x[0]) : conf1})
                    if x[1] not in self.finalPatterns:
                        self.finalPatterns[x[1]] = {x[0]:conf}
            if len(x) > 2:
                self.generation(x)


if __name__ == '__main__':
    data = {'c':2, 'cb':2, 'b':5,
            'bd':3, 'ba':3, 'd':5, 'da':3, 'a':5}
    ap = RuleMiner(data, "confidence", 1)
    ap.RuleMiner()