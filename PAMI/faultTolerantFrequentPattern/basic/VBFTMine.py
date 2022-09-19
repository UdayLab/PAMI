#  Copyright (C)  2021 Rage Uday Kiran
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

from PAMI.faultTolerantFrequentPattern.basic import abstract as _ab

class FTApriori(_ab._faultTolerantFrequentPatterns):
    """
        FT-Apriori is one of the fundamental algorithm to discover fault tolerant frequent patterns in a transactional database.
        This program employs apriori property (or downward closure property) to  reduce the search space effectively.

        Reference:
        ----------
            Pei, Jian & Tung, Anthony & Han, Jiawei. (2001). Fault-Tolerant Frequent Pattern Mining: Problems and Challenges.


        Attributes:
        ----------
                iFile : str
                    Input file name or path of the input file
                oFile : str
                    Name of the output file or the path of output file
                minSup: float or int or str
                    The user can specify minSup either in count or proportion of database size.
                    If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
                    Otherwise, it will be treated as float.
                    Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
                sep : str
                    This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
                    However, the users can override their default separator.
                startTime:float
                    To record the start time of the mining process
                endTime:float
                    To record the completion time of the mining process
                finalPatterns: dict
                    Storing the complete set of patterns in a dictionary variable
                memoryUSS : float
                    To store the total amount of USS memory consumed by the program
                memoryRSS : float
                    To store the total amount of RSS memory consumed by the program
                Database : list
                    To store the transactions of a database in list

            Methods:
            -------
                startMine()
                    Mining process will start from here
                getPatterns()
                    Complete set of patterns will be retrieved with this function
                save(oFile)
                    Complete set of frequent patterns will be loaded in to a output file
                getPatternsAsDataFrame()
                    Complete set of frequent patterns will be loaded in to a dataframe
                getMemoryUSS()
                    Total amount of USS memory consumed by the mining process will be retrieved from this function
                getMemoryRSS()
                    Total amount of RSS memory consumed by the mining process will be retrieved from this function
                getRuntime()
                    Total amount of runtime taken by the mining process will be retrieved from this function
                candidateToFrequent(candidateList)
                    Generates frequent patterns from the candidate patterns
                frequentToCandidate(frequentList, length)
                    Generates candidate patterns from the frequent patterns


            Executing the code on terminal:
            -------------------------------

                Format:
                ------
                    python3 FTApriori.py <inputFile> <outputFile> <minSup> <itemSup> <minLength> <faultTolerance>

                Examples:
                ---------
                    python3 FTApriori.py sampleDB.txt patterns.txt 10.0 3.0 3 1  (minSup will be considered in times of minSup and count of database transactions)

                    python3 FTApriori.py sampleDB.txt patterns.txt 10  3 2 1    (minSup will be considered in support count or frequency)


            Sample run of the importing code:
            ---------------------------------

                import PAMI.faultTolerantFrequentPattern.basic.FTApriori as alg

                obj = alg.FTApriori(iFile, minSup, itemSup, minLength, faultTolerance)

                obj.startMine()

                faultTolerantFrequentPatterns = obj.getPatterns()

                print("Total number of Fault Tolerant Frequent Patterns:", len(faultTolerantFrequentPatterns))

                obj.save(oFile)

                Df = obj.getPatternInDataFrame()

                print("Total Memory in USS:", obj.getMemoryUSS())

                print("Total Memory in RSS", obj.getMemoryRSS())

                print("Total ExecutionTime in seconds:", obj.getRuntime())

            Credits:
            --------
                The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.

        """

    _minSup = float()
    _itemSup = float()
    _minLength = int()
    _faultTolerance = int()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _mapSupport = {}

    def _creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable


        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            temp = []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                temp = self._iFile['Transactions'].tolist()

            for k in temp:
                self._Database.append(set(k))
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(set(temp))
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(set(temp))
                except IOError:
                    print("File Not Found")
                    quit()

    def _convert(self, value):
        """
        To convert the user specified minSup value

        :param value: user specified minSup value

        :return: converted type
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self._Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self._Database) * value)
            else:
                value = int(value)
        return value



import sys
import time
import resource
import math
import numpy as np
path=sys.argv[1]
output=sys.argv[2]
itemSup=int(sys.argv[3])
minSup=int(sys.argv[4])
faultTolerance=int(sys.argv[5])
maxPer=int(sys.argv[6])
Vector={}
lno=0
plist=[]
transactions=[]
items=[]
def Per_Sup(tids):
    per=0
    cur=0
    sup=0
    for i in range(len(tids)):
        if tids[i]==1:
            per=max(per,cur-i)
            if per>maxPer:
                return [0,0]
            cur=i
            sup+=1
    per=max(per,lno-cur)
    return [sup,per]
def Count(tids):
    count=0
    for i in tids:
        if i==1:
            count+=1
    return count
         
with open(path,'r') as f:
    for line in f:
        lno+=1
        l=line.split()
        transactions.append(l)
        for i in l[1:]:
            if i not in plist:
                plist.append(i)
for i in transactions:
    for j in plist:
        count=0
        if j in i:
            count=1
        if j in Vector:
            Vector[j].append(count)
        else:
            Vector[j]=[count]
for x,y in Vector.items():
    v=Count(y)
    if v>=itemSup:
        items.append(x)
        
def save(prefix,suffix,tidsetx):
        if(prefix==None):
            prefix=suffix
        else:
            prefix=prefix+suffix
        prefix=list(set(prefix))
        prefix.sort()
        val=Count(tidsetx)
        print(prefix,val)
        
def processEquivalenceClass(prefix,itemsets,tidsets):
        if(len(itemsets)==1):
            i=itemsets[0]
            tidi=tidsets[0]
            save(prefix,[i],tidi)
            return
        for i in range(len(itemsets)):
            itemx=itemsets[i]
            if(itemx==None):
                continue
            tidsetx=tidsets[i]
            classItemsets=[]
            classtidsets=[]
            itemsetx=[itemx]
            for j in range(i+1,len(itemsets)):
                itemj=itemsets[j]
                tidsetj=tidsets[j]
                y=list(np.array(tidsetx) & np.array(tidsetj))
                total=Count(y)
                if total>=itemSup:
                    classItemsets.append(itemj)
                    classtidsets.append(y)
            if(len(classItemsets)>0):
                newprefix=list(set(itemsetx))+prefix
                processEquivalenceClass(newprefix, classItemsets,classtidsets,classItemSets)
            save(prefix,list(set(itemsetx)),itemTidsi)

def Mine(plist):
    for i in range(len(plist)):
            itemx=plist[i]
            tidsetx=Vector[itemx]
            itemsetx=[itemx]
            itemsets=[]
            itemSets=[]
            tidsets=[]
            for j in range(i+1,len(plist)):
                itemj=plist[j]
                tidsetj=Vector[itemj]
                y1=list(np.array(tidsetx) & np.array(tidsetj))
                total=Count(y1)
                if total>=itemSup:
                    itemsets.append(itemj)
                    tidsets.append(y1)
            if(len(itemsets)>0):
                processEquivalenceClass(itemsetx,itemsets,tidsets)
            save(None,itemsetx,tidsetx)
    
               
Mine(items)
