import numpy as np
import pandas as pd


def csvToBitInteger(file, output, sep = "\t"):

    fileData = {}
    rename = {}
    counter = 0
    maxTS = 0


    with open(file,'r') as f:
        
        for line in f:
            line = line.strip().split(sep)
            timestamp = int(line[0])
            if timestamp > maxTS:
                maxTS = timestamp

            line = line[1:]

            for item in line:
                if item not in rename:
                    rename[item] = counter
                    counter += 1
                if rename[item] not in fileData:
                    fileData[rename[item]] = [timestamp]
                else:
                    fileData[rename[item]].append(timestamp)

    fileData = dict(sorted(fileData.items(), key = lambda x: len(x[1])))


    newRep = {}
    # arraySize = maxTS // 32 + 1

    # print("Filedata:",len(fileData))
    for k,v in fileData.items():
        bitRep = np.zeros(maxTS + 1, dtype=np.uint32)
        for i in range(len(v)):
            bitRep[v[i]] = 1
        newRep[k] = bitRep
        # print(k,":",len(v))

    # print("Creating dataframe...")
    df = pd.DataFrame(newRep)
    # print("Transposing dataframe...")
    df = df.T

    cols = []

    for i in range(len(df.columns)):
        cols.append(str(i))

    df.columns = cols

    # print("Writing to parquet...")
    df.to_parquet(output)

