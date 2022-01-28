import os
import csv
import time
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import psutil


class cudaEclatGCT:
    __time = 0
    __memRSS = 0
    __memUSS = 0
    __GPU_MEM = 0
    filePath = ""
    sep = ""
    minSup = 0
    Patterns = {}

    def __init__(self, filePath, sep, minSup):
        self.filePath = filePath
        self.sep = sep
        self.minSup = minSup
        self.__time = 0
        self.__memRSS = 0
        self.__memUSS = 0

    def read_data(self, data_path, sep):
        data = []
        if not os.path.isfile(data_path):
            raise ValueError('Invalid data path.' + data_path)
        with open(data_path, 'r') as f:
            file = csv.reader(f, delimiter=sep, quotechar='\r')
            lineNo = 1
            for row in file:
                data.append([str(item) for item in row if item != ''])
                lineNo += 1
        return data, lineNo

    def write_result(self, result, result_path):
        file = open(result_path, 'w')
        for itemset, support in result.items():
            file.write(str(itemset) + ' : ' + str(support) + '\n')
        file.close()

    def compute_vertical_bitvector_data(self, data):
        #---build item to idx mapping---#
        idx = 0
        item2idx = {}
        for transaction in data:
            for item in transaction:
                if not item in item2idx:
                    item2idx[item] = idx
                    idx += 1
        idx2item = {idx: str(int(item)) for item, idx in item2idx.items()}
        #---build vertical data---#
        vb_data = np.zeros((len(item2idx), len(data)), dtype=np.uint16)
        for trans_id, transaction in enumerate(data):
            for item in transaction:
                vb_data[item2idx[item], trans_id] = 1
        vb_data = gpuarray.to_gpu(vb_data.astype(np.uint16))
        return vb_data, idx2item

    def get_time(self):
        return self.__time

    def get_memRSS(self):
        return self.__memRSS

    def get_memUSS(self):
        return self.__memUSS

    def get_GPU_MEM(self):
        return self.__GPU_MEM

    def get_Patterns(self):
        return self.Patterns

    def get_numberOfPatterns(self):
        return len(self.Patterns)

    def eclat(self, basePattern, final, vb_data, idx2item, item2idx):
        newBasePattern = []
        for i in range(0, len(basePattern)):
            item1 = basePattern[i]
            i1_list = item1.split()
            for j in range(i + 1, len(basePattern)):
                item2 = basePattern[j]
                i2_list = item2.split()
                if i1_list[:-1] == i2_list[:-1]:
                    unionOfKey = list(set(i1_list) | set(i2_list))
                    unionOfKey.sort()
                    valueList = []
                    for key in unionOfKey:
                        valueList.append(item2idx[key])
                    total = vb_data[valueList[0]]
                    for k in range(1, len(valueList)):
                        total = total.__mul__(vb_data[valueList[k]])
                    support = gpuarray.sum(total).get()
                    if support >= self.minSup:
                        newBasePattern.append(" ".join(unionOfKey))
                        final[" ".join(unionOfKey)] = support

        if len(newBasePattern) > 0:
            self.eclat(newBasePattern, final, vb_data, idx2item, item2idx)

    def startMine(self):
        startTime = time.time()
        basePattern = []
        final = {}

        data, lineNo = self.read_data(self.filePath, self.sep)
        vb_data, idx2item = self.compute_vertical_bitvector_data(data)
        if self.minSup < 1:
            self.minSup = int(lineNo * self.minSup)

        for i in range(len(vb_data)):
            if gpuarray.sum(vb_data[i]).get() >= self.minSup:
                basePattern.append(idx2item[i])
                final[idx2item[i]] = gpuarray.sum(vb_data[i]).get()

        # reverse idx2item
        item2idx = {idx2item[i]: i for i in idx2item}
        self.eclat(basePattern, final, vb_data, idx2item, item2idx)
        self.__time = time.time() - startTime
        self.__memRSS = psutil.Process(os.getpid()).memory_info().rss
        self.__memUSS = psutil.Process(os.getpid()).memory_full_info().uss
        self.Patterns = final
        self.__GPU_MEM = vb_data.nbytes


if __name__ == "__main__":
    filePath = "datasets\\transactional_T10I4D100K.csv"
    # filePath = "file.txt"
    sep = "\t"
    support = 500
    cudaEclatGCT = cudaEclatGCT(filePath, sep, support)
    cudaEclatGCT.startMine()
    print("Time: ", cudaEclatGCT.get_time())
    print("Memory RSS: ", cudaEclatGCT.get_memRSS())
    print("Memory USS: ", cudaEclatGCT.get_memUSS())
    print("GPU MEM: ", cudaEclatGCT.get_GPU_MEM())
    print("Patterns: ", cudaEclatGCT.get_numberOfPatterns())
