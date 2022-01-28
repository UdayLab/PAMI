import os
import csv
import time
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import psutil
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda

deviceIntersection = SourceModule("""
    __global__ void intersection(int *compareThis, int *compareThat, int *resultStart,
                                 int *values, int *result, int resultX, int resultY){
        const int tidX = blockIdx.x * blockDim.x + threadIdx.x;
        const int tidY = blockIdx.y * blockDim.y + threadIdx.y; 
        int resultIndex = resultStart[tidX] + tidY;
        
        // ignore if tidX or tidY is out of bounds or if the value comparing with is 0
        if (tidX > resultX-1 || tidY > resultY-1 || values[compareThis[tidX] + tidY] == 0) return;

        for (int i = 0; i < resultY; i++){
            if ( values[compareThat[tidX] + i] == 0) return;
            if ( values[compareThis[tidX] + tidY] == values[compareThat[tidX] + i]){
                result[resultIndex] = values[compareThis[tidX] + tidY];
                return;
            }
        }

        //result[resultIndex] = values[compareThis[tidX] + tidY];

    }

"""
                                  )


class cudaAprioriTID:
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

    def _readFile(self, fileName, separator):
        """
        Reads a file and stores the data in a dictionary

        Args:
            fileName: string
            separator: string

        Returns:
            dictionary: dictionary
        """
        file = open(fileName, 'r')
        dictionary = {}
        lineNumber = 1
        for line in file:
            line = line.strip()
            line = line.split(separator)
            for i in range(len(line)):
                if line[i] in dictionary:
                    dictionary[line[i]].append(lineNumber)
                else:
                    dictionary[line[i]] = [lineNumber]
            lineNumber += 1

        # sort dictionary by size of values
        dictionary = dict(
            sorted(dictionary.items(), key=lambda x: len(x[1]), reverse=True))
        return dictionary, lineNumber

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

    def startMine(self):
        dev_Intersection = deviceIntersection.get_function("intersection")
        startTime = time.time()
        final = {}

        data, lineNo = self._readFile(self.filePath, self.sep)
        if self.minSup < 1:
            self.minSup = int(lineNo * self.minSup)

        data = dict(filter(lambda x: len(x[1]) >= self.minSup, data.items()))
        for key, value in data.items():
            final[key] = len(value)

        while len(data) > 1:
            # sort data by size of values
            data = dict(
                sorted(data.items(), key=lambda x: len(x[1]), reverse=True))

            values = list(data.values())
            maxLength = values[0]
            for i in range(1, len(values)):
                while len(values[i]) != len(maxLength):
                    values[i].append(0)

            values = np.array(values)
            resultSize = 0

            compareThis = []
            compareThat = []
            resultStart = []
            counter = 0

            for i in range(len(values)):
                for j in range(i+1, len(values)):
                    resultSize += 1
                    compareThis.append(i*len(maxLength))
                    compareThat.append(j*len(maxLength))
                    resultStart.append(counter)
                    counter += len(maxLength)
            result = np.zeros((resultSize, len(maxLength)), dtype=np.int32)

            # convert all to uint32
            compareThis = np.array(compareThis, dtype=np.uint32)
            compareThat = np.array(compareThat, dtype=np.uint32)
            resultStart = np.array(resultStart, dtype=np.uint32)
            values = np.array(values, dtype=np.uint32)
            result = np.array(result, dtype=np.uint32)

            # allocate memory on GPU
            compareThis_gpu = cuda.mem_alloc(compareThis.nbytes)
            compareThat_gpu = cuda.mem_alloc(compareThat.nbytes)
            resultStart_gpu = cuda.mem_alloc(resultStart.nbytes)
            values_gpu = cuda.mem_alloc(values.nbytes)
            result_gpu = cuda.mem_alloc(result.nbytes)

            # add all nbytes to GPU_MEM
            sumBytes = compareThis.nbytes + compareThat.nbytes + resultStart.nbytes + values.nbytes + result.nbytes
            if sumBytes > self.__GPU_MEM:
                self.__GPU_MEM = sumBytes

            # copy data to GPU
            cuda.memcpy_htod(compareThis_gpu, compareThis)
            cuda.memcpy_htod(compareThat_gpu, compareThat)
            cuda.memcpy_htod(resultStart_gpu, resultStart)
            cuda.memcpy_htod(values_gpu, values)
            cuda.memcpy_htod(result_gpu, result)

            blockDim = (32, 32, 1)
            gridDim = (resultSize//32 + 1, len(maxLength)//32 + 1, 1)

            dev_Intersection(compareThis_gpu, compareThat_gpu,
                             resultStart_gpu, values_gpu, result_gpu,
                             np.uint32(resultSize), np.uint32(len(maxLength)),
                             block=blockDim, grid=gridDim)

            # copy data back to CPU
            cuda.Context.synchronize()
            cuda.memcpy_dtoh(result, result_gpu)

            # free GPU memory
            cuda.DeviceAllocation.free(compareThis_gpu)
            cuda.DeviceAllocation.free(compareThat_gpu)
            cuda.DeviceAllocation.free(resultStart_gpu)
            cuda.DeviceAllocation.free(values_gpu)
            cuda.DeviceAllocation.free(result_gpu)

            keys = list(data.keys())
            # convert all to string and add " "
            for i in range(len(keys)):
                keys[i] = str(keys[i]) + " "
            data = {}
            index = 0
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    newResult = list(sorted(set(result[index])))
                    newResult = list(filter(lambda x: x > 0, newResult))
                    if len(newResult) >= self.minSup:
                        keyI = keys[i].split()
                        keyJ = keys[j].split()
                        combinedKey = " ".join(list(str(x) for x in (
                            sorted(int(x) for x in (set(keyI) | set(keyJ))))))
                        if combinedKey not in final:
                            data[combinedKey] = newResult
                            final[combinedKey] = len(newResult)
                    index += 1


        self.__time = time.time() - startTime
        self.__memRSS = psutil.Process(os.getpid()).memory_info().rss
        self.__memUSS = psutil.Process(os.getpid()).memory_full_info().uss
        self.Patterns = final


if __name__ == "__main__":
    filePath = "datasets\\transactional_T10I4D100K.csv"
    sep = "\t"
    support = 500
    cudaAprioriTID = cudaAprioriTID(filePath, sep, support)
    cudaAprioriTID.startMine()
    print("Time: ", cudaAprioriTID.get_time())
    print("Memory RSS: ", cudaAprioriTID.get_memRSS())
    print("Memory USS: ", cudaAprioriTID.get_memUSS())
    print("GPU MEM: ", cudaAprioriTID.get_GPU_MEM())
    print("Patterns: ", cudaAprioriTID.get_numberOfPatterns())
