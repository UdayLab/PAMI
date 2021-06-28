import rsfp as alg

obj = alg.rsfp("sampleTDB.txt","3",0.8)

obj.startMine()

frequentPatterns = obj.getFrequentPatterns()
#print(frequentPatterns)

print("Total number of Frequent Patterns:", len(frequentPatterns))

obj.storePatternsInFile("oFile")

Df = obj.getPatternsInDataFrame()

memUSS = obj.getMemoryUSS()

print("Total Memory in USS:", memUSS)

memRSS = obj.getMemoryRSS()

print("Total Memory in RSS", memRSS)

run = obj.getRuntime()

print("Total ExecutionTime in seconds:", run)
