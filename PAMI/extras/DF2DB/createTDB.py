import pandas as pd
from PAMI.frequentPattern.basic import FPGrowth as fp


class createTDB:

    def __init__(self, df, threshold):
        self._df = df
        self._threshold = int(threshold)
        self._items = []
        self._updatedItems = []

    def createTDB(self):
        """
            Create transactional data base

            :returning a transactional database as DataFrame
        """
        i = self._df.columns.values.tolist()
        if 'sid' in i:
            self._items = self._df['sid'].tolist()
        for i in self._items:
            i = i.split()
            self._updatedItems.append([j for j in i if int(j) > self._threshold])

    def savePatterns(self, outFile):
        """
            Complete set of frequent patterns will be loaded in to a output file

            :param outFile: name of the output file

            :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x in self._updatedItems:
            s = str()
            for j in x:
                s = s + j + " "
            writer.write("%s \n" % s)


if __name__ == '__main__':
    a = createTDB('DataFrame', "1204150")
    a.createTDB()
    a.savePatterns('output.txt')
    ap = fp.FPGrowth('output.txt', 500, ' ')
    ap.startMine()
    Patterns = ap.getPatterns()
    print("Total number of Frequent Patterns:", len(Patterns))
    ap.savePatterns('fpoutput.txt')
    memUSS = ap.getMemoryUSS()
    print("Total Memory in USS:", memUSS)
    memRSS = ap.getMemoryRSS()
    print("Total Memory in RSS", memRSS)
    run = ap.getRuntime()
    print("Total ExecutionTime in ms:", run)


