import random as _rd
import sys as _sys


class createSyntheticUtility:
    """
        This class create synthetic utility database. 

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

        Nethods:
        --------
        getUtilitylDatabase(outputFile)
            Create utility database from DataFrame and store into outputFile

        Credits:
        ---------
             The complete program was written by  P.Likhitha   under the supervision of Professor Rage Uday Kiran.

    """
    
    def __init__(self, transactions, items, maxUtilRange, avgTransaction):
        self._transactions = transactions
        self._items = items
        self._maxUtilrange = maxUtilRange
        self._avgTransaction = avgTransaction
    
    def createUtilityDatabase(self, outputFile):
        """
        create transactional database and return outputFileName
        :param outputFile: file name or path to store database
        :type outputFile: str
        :return: outputFile name
        """
        writer = open(outputFile, 'w+')
        for i in range(self._transactions):
            length = _rd.randint(1, self._avgTransaction + 20)
            st = str()
            st1 = str()
            su = []
            for i in range(length):
                item = _rd.randint(1, self._items)
                utility = _rd.randint(1, self._maxUtilrange) 
                st = st + str(item) + '\t'
                su.append(utility)
                st1 = st1 + str(utility) + '\t'
            summation = sum([i for i in su])
            st = st + ":" + str(summation) + ":"
            writer.write("%s" % st)
            writer.write("%s \n" % st1)
            
if __name__ == "__main__":
    _ap = str()
    _ap = createSyntheticUtility(100000, 870, 100, 10)
    _ap.createUtilityDatabase("T10_util.txt")
else:
    print("Error! The number of input parameters do not match the total number of parameters provided")