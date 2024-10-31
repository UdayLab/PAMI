#csvParquet is a code used to convert temporal and utility types into sparse and dense format.
#
#**Importing this algorithm into a python program**
#--------------------------------------------------------
#
#             from PAMI.extras.csvParquet import csvParquet as cp
#
#             obj = cp.csvParquet(iFile, "\t", " ", " " )
#
#             obj.save()
#
#             obj.csvParquet("FileName") # To generate file in form of sparse or dense
#
#             obj.parquetFormat("FileName") # To generate file in form of sparse or dense
#





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
import numpy as np

class CSV2Parquet():
    """

    :Description:   csvParquet is a code used to convert temporal and utility types into sparse and dense format

    :param  iFile: str :
                   Name of the Input file
    :param  sep: str :
                    This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.

    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

            from PAMI.extras.csvParquet import csvParquet as cp

            obj = cp.csvParquet(File, "\t", " ", " " )

            obj.save()

            obj.csvParquet("FileName") # To generate file in form of sparse or dense

            obj.parquetFormat("FileName") # To generate file in form of sparse or dense
    """

    def __init__(self, iFile: str, sep: str='\t'):
        self._iFile = iFile
        self._sep = sep

def CSV2Parquet(csv_file: str, sep: str, inputType: str, outputType: str) -> None:
    inputTypes = ["temporal", "utility"]
    outputTypes = ["sparse", "dense"]

    inputType = inputType.lower()

    error = False

    if inputType not in inputTypes:
        print("Input type must be one of: " + str(inputTypes))
        error = True

    outputType = outputType.lower()
    if outputType not in outputTypes:
        print("Output type must be one of: " + str(outputTypes))
        error = True
    
    if error:
        return
    
    file = csv_file.split(".")
    parquet_file = file[0] + ".parquet"
    dict_file = file[0] + ".dict"
    
    if inputType == "temporal":
        conversion = {}
        conNum = 1

        file = []
        indexes = []
        for line in open(csv_file):
            # first item is the index
            line = line.strip().split(sep)
            indexes.append(int(line[0]))
            # file.append([int(i) for i in line[1:]])
            temp = []
            for i in line[1:]:
                if i not in conversion:
                    conversion[i] = conNum
                    conNum += 1
                temp.append(conversion[i])
            file.append(temp)

        if outputType == 'dense':
            sparseList = []
            indx = []

            for i in range(len(file)):
                for j in range(len(file[i])):
                    sparseList.append(file[i][j])
                    indx.append(indexes[i])

            columns = [str(i) for i in range(1, 2)]
            df = pd.DataFrame(sparseList, columns=columns, index=indx)
            df.to_parquet(parquet_file, engine='pyarrow')


        elif outputType == 'sparse':
            maxLen = max([len(i) for i in file])

            for i in range(len(file)):
                if len(file[i]) < maxLen:
                    file[i].extend([-1] * (maxLen - len(file[i])))

            columns = [str(i) for i in range(1, maxLen+1)]
            df = pd.DataFrame(file, columns=columns, index=indexes)
            df.to_parquet(parquet_file, engine='pyarrow')

        

    elif inputType == "utility":
        conversion = {}
        conNum = 1

        file = []

        indexes = 1
        maxLen = 0

        for line in open(csv_file):
            line = line.strip().split(":")
            items = line[0].split(sep)
            values = line[2].split(sep)
            values = [float(i) for i in values]
            temp = []
            for i in items:
                if i not in conversion:
                    conversion[i] = conNum
                    conNum += 1
                temp.append(conversion[i])
            temp = [int(i) for i in temp]
            file.append([indexes, temp, values])
            maxLen = max(maxLen, len(values))
            indexes += 1


        if outputType == 'dense':
            newFile = []
            indexes = []

            for i in range(len(file)):
                for j in range(len(file[i][1])):
                    newFile.append([file[i][1][j], file[i][2][j]])
                    indexes.append(file[i][0])
            
            columns = [str(i) for i in range(1, 3)]

            df = pd.DataFrame(newFile, columns=columns, index=indexes)
            df.to_parquet(parquet_file, engine='pyarrow')

        elif outputType == 'sparse':
            newFile = []
            indexes = []
            for i in range(len(file)):
                newFile.append([])
                for j in range(len(file[i][1])):
                    newFile[-1].append(file[i][1][j])
                    newFile[-1].append(file[i][2][j])
                indexes.append(file[i][0])

            columns = [str(i) for i in range(1, (maxLen*2) + 1)]
            df = pd.DataFrame(newFile, index=indexes, columns=columns)
            df.to_parquet(parquet_file, engine='pyarrow')

            
    with open(dict_file, 'w') as f:
            for key in conversion.keys():
                f.write("%s->%s\n"%(key,conversion[key]))


def parquetFormat(file: str, sep: str, inputType: str, outputType: str) -> None:
    inputTypes = ["temporal", "utility"]
    outputTypes = ["sparse", "dense"]

    inputType = inputType.lower()

    error = False

    if inputType not in inputTypes:
        print("Input type must be one of: " + str(inputTypes))
        error = True

    outputType = outputType.lower()
    if outputType not in outputTypes:
        print("Output type must be one of: " + str(outputTypes))
        error = True

    if error:
        return
    
    df = pd.read_parquet(file, engine='pyarrow')

    if inputType == "temporal":

        if outputType == "dense":
            ndf = {}
            indexes = df.index.tolist()
            # # ndf = df.values.tolist()
            for i in range(len(df)):
                # ndf[i] = [j for j in ndf[i] if j != -1]
                if indexes[i] not in ndf:
                    ndf[indexes[i]] = []
                ndf[indexes[i]].append([j for j in df.iloc[i] if j != -1])

            sparseList = []
            indx = []

            indexes = list(ndf.keys())
            filed = [x[0] for x in list(ndf.values())]

            for i in range(len(filed)):
                for j in range(len(filed[i])):
                    sparseList.append(filed[i][j])
                    indx.append(indexes[i])

            columns = [str(i) for i in range(1, 2)]
            df = pd.DataFrame(sparseList, columns=columns, index=indx)
            df.to_parquet(file, engine='pyarrow')
        
        if outputType == "sparse":
            ndf = {}
            indexes = df.index.tolist()


            for i in range(len(df)):
                if indexes[i] not in ndf:
                    ndf[indexes[i]] = []
                for j in range(len(df.iloc[i])):
                    # ndf[indexes[i]].append(int(df.iloc[i,j]))
                    if df.iloc[i,j] != -1:
                        ndf[indexes[i]].append(int(df.iloc[i,j]))
                    else:
                        break

            
            indexes = list(ndf.keys())
            nfile = list(ndf.values())

            maxLen = max([len(i) for i in nfile])

            for i in range(len(nfile)):
                if len(nfile[i]) < maxLen:
                    nfile[i].extend([-1] * (maxLen - len(nfile[i])))

            columns = [str(i) for i in range(1, maxLen+1)]
            df = pd.DataFrame(nfile, columns=columns, index=indexes)
            df.to_parquet(file, engine='pyarrow')

    
    if inputType == "utility":
        ndf = []

        if outputType == "sparse":
            ndf = {}

            indexes = df.index.tolist()

            for i in range(len(df)):
                index = indexes[i]
                if index not in ndf:
                    ndf[index] = []
                for j in range(len(df.iloc[i])):
                    if j % 2 == 0:
                        ndf[index].append(int(df.iloc[i,j]))
                    else:
                        ndf[index].append(float(df.iloc[i,j]))

            indexes = list(ndf.keys())
            nfile = list(ndf.values())

            maxLen = max([len(i) for i in nfile])

            
            columns = [str(i) for i in range(1, (maxLen+1))]
            df = pd.DataFrame(nfile, columns=columns, index=indexes)
            df.to_parquet(file, engine='pyarrow')
            

        if outputType == "dense":
            ndf = {}

            indexes = df.index.tolist()

            for i in range(len(df)):
                index = indexes[i]
                if index not in ndf:
                    ndf[index] = [[],[]]

                numbers = df.iloc[i,0::2].dropna().tolist()
                values = df.iloc[i,1::2].dropna().tolist()

                zipped = list(zip(numbers, values))
                for i in range(len(zipped)):
                    ndf[index][0].append(int(zipped[i][0]))
                    ndf[index][1].append(float(zipped[i][1]))

            indexes = list(ndf.keys())
            nfile = list(ndf.values())

            newFile = []
            iindexes = []

            for i in range(len(nfile)):
                for j in range(len(nfile[i][1])):
                    newFile.append([nfile[i][0][j], nfile[i][1][j]])
                    iindexes.append(indexes[i])

            lens = [len(i) for i in newFile]
            
            columns = [str(i) for i in range(1, 3)]

            # nums = [i[0] for i in newFile]
            # vals = [i[1] for i in newFile]

            df = pd.DataFrame(newFile, columns=columns, index=iindexes)
            # df = pd.DataFrame([nums, vals], index=iindexes)
            # print(df)
            df.to_parquet(file, engine='pyarrow')
