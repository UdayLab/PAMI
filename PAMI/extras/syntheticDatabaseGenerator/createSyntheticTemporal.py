import random as _rd
import sys as _sys
class createSyntheticTemporal:
    """
        This class create synthetic temporal database. 

        Attribute:
        ----------
        transactions : pandas.DataFrame
            No of transactions
        items : int or float
            No of items
        avgTransaction : str
            The length of average transaction
        outputFile: str
            Name of the output file.

        Methods:
        --------
        getTemporallDatabase(outputFile)
            Create temporal database from DataFrame and store into outputFile

        Credits:
        ---------
             The complete program was written by  P.Likhitha   under the supervision of Professor Rage Uday Kiran.


    """
    
    def __init__(self, transactions, items, avgTransaction):
        self._transactions = transactions
        self._items = items
        self._avgTransaction = avgTransaction
    
    def createTemporalDatabase(self, outputFile):
        """
        create transactional database and return outputFileName
        :param outputFile: file name or path to store database
        :type outputFile: str
        :return: outputFile name
        """
        count = 1
        writer = open(outputFile, 'w+')
        for i in range(self._transactions):
            length = _rd.randint(1, self._avgTransaction + 20)
            st = str(count) + '\t'
            for i in range(length):
                item = _rd.randint(1, self._items)
                st = st + str(item) + '\t'
            writer.write("%s \n" % st)
            count += 1
            
if __name__ == "__main__":
    _ap = str()
    _ap = createSyntheticTemporal(100000, 870, 10)
    _ap.createTemporalDatabase("temporal_T10.txt")
else:
    print("Error! The number of input parameters do not match the total number of parameters provided")