import sys as _sys
import pandas as _pd
import validators as _validators
import statistics as _statistics
from urllib.request import urlopen as _urlopen

class usingBeta():

    _iFile = ' '
    _sd = int()
    _sep = str()
    _threshold = int()
    _finalPatterns = {}

    def __init__(self, iFile, threshold, sep):
        self._iFile = iFile
        self._threshold = threshold
        self._sep = sep

    def _creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable
        """
        self._Database = []
        self._mapSupport = {}
        if isinstance(self._iFile, _pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()

        if isinstance(self._iFile, str):
            if _validators.url(self._iFile):
                data = _urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r') as f:
                        for line in f:
                            self._lno += 1
                            splitter = [i.rstrip() for i in line.split(self._sep)]
                            splitter = [x for x in splitter if x]
                            self._Database.append(splitter)
                except IOError:
                    print("File Not Found")

    def _creatingFrequentItems(self):
        """
        This function creates frequent items from _database.
        :return: frequentTidData that stores frequent items and their tid list.
        """
        tidData = {}
        self._lno = 0
        for transaction in self._Database:
            self._lno = self._lno + 1
            for item in transaction:
                if item not in tidData:
                    tidData[item] = [self._lno]
                else:
                    tidData[item].append(self._lno)
        mini = min([len(k) for k in tidData.values()])
        sd = _statistics.stdev([len(k) for k in tidData.values()])
        frequentTidData = {k: len(v) - sd for k, v in tidData.items()}
        return mini, frequentTidData

    def caculateMIS(self):
        self._creatingItemSets()
        mini, frequentItems = self._creatingFrequentItems()
        for x, y in frequentItems.items():
            if y < self._threshold:
                self._finalPatterns[x] = mini
            else:
                self._finalPatterns[x] = y

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe
        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b])
            dataFrame = _pd.DataFrame(data, columns=['Items', 'MIS'])
        return dataFrame

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = x + ":" + str(y)
            writer.write("%s \n" % patternsAndSupport)

if __name__ == '__main__':
    cd = usingBeta("sample.txt", 10, ' ')
    cd.caculateMIS()
    cd.save('output.txt')