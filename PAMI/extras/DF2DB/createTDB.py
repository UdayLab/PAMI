# createTDB in this code  we will create transactional Database.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.DF2DB import createTDB as ct
#
#     obj = ct.createTDB(idf, ">=", 16)
#
#     obj.save(oFile)
#
#

import sys

__copyright__ = """
 Copyright (C)  2021 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd


class createTDB:
    """
    :Description: This class will create Transactional database.

    :param df: It represents the dataframe

    :type df: list

    :param threshold : It is the threshold value of all item.

    :type threshold: int or float



        **Importing this algorithm into a python program**
        --------------------------------------------------------
        .. code-block:: python

                    from PAMI.frequentPattern.basic import FPGrowth as fp

                    obj = fp.createTDB(idf, ">=" )

                    memUSS = obj.getMemoryUSS()

                    print("Total Memory in USS:", memUSS)

                    memRSS = obj.getMemoryRSS()

                    print("Total Memory in RSS", memRSS)

                    run = obj.getRuntime()

                    print("Total ExecutionTime in seconds:", run)




        """
    __startTime = float()
    __endTime = float()
    __memoryUSS = float()
    __memoryRSS = float()
    __Database = []
    __finalPatterns = {}

    def __init__(self, df, threshold):
        self._df = df
        self._threshold = int(threshold)
        self._items = []
        self._updatedItems = []

    def createTDB(self):
        """
            :Description:  To Create transactional database


        """
        i = self._df.columns.values.tolist()
        if 'sid' in i:
            self._items = self._df['sid'].tolist()
        for i in self._items:
            i = i.split()
            self._updatedItems.append([j for j in i if int(j) > self._threshold])

    def save(self, outFile):
        """
            Complete set of frequent patterns will be loaded in to an output file

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
    a = createTDB(sys.argv[1], sys.argv[3])
    a.createTDB()
    a.save(sys.argv[2])