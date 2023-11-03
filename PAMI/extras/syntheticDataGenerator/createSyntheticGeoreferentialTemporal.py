import random as _rd
import sys as _sys


class createGeoreferentialTemporalDatabase:
    """
        This class create synthetic geo-referential temporal database. 

        Attribute:
        ----------
        totalTransactions : int
            No of transactions
        noOfItems : int or float
            No of items
        avgTransactionLength : str
            The length of average transaction
        outputFile: str
            Name of the output file.

        Methods:
        --------
        createGeoreferentialTemporalDatabase(outputFile)
            Create geo-referential temporal database and store into outputFile

        Credits:
        ---------
             The complete program was written by  P.Likhitha   under the supervision of Professor Rage Uday Kiran.

    """
    
    def __init__(self, transactions: int, items: int, avgTransaction: int) -> None:
        self._totalTransactions = transactions
        self._noOfItems = items
        self._avgTransactionLength = avgTransaction
    
    def createGeoreferentialTemporalDatabase(self, outputFile: str) -> None:
        """
        create transactional database and return outputFileName
        :param outputFile: file name or path to store database
        :type outputFile: str
        :return: outputFile name
        """
        writer = open(outputFile, 'w+')
        items = []
        count = 1
        for i in range(self._noOfItems):
            lat = _rd.randint(1, self._noOfItems)
            lon = _rd.randint(1, self._noOfItems)
            if lat == lon:
                lon = _rd.randint(1, self._noOfItems)
            stt = '(' + str(lat) + ' ' + str(lon) + ')'
            items.append(stt)
        for i in range(self._totalTransactions):
            length = _rd.randint(1, self._avgTransactionLength + 20)
            st = str(count)
            for i in range(length):
                rd = _rd.randint(0, len(items) - 1)
                item = items[rd]
                st = st + str(item) + '\t'
            writer.write("%s \n" % st)
            count += 1
            
if __name__ == "__main__":
    _ap = str()
    _ap = createSyntheticGeoreferentialTemporal(100000, 870, 10)
    _ap.createGeoreferentialTemporalDatabase("T10_geo_temp.txt")
else:
    print("Error! The number of input parameters do not match the total number of parameters provided")
