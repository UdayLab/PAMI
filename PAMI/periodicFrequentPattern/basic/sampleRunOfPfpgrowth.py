import PFPGrowth as alg

obj = alg.PFPGrowth("sampleTDB.txt", "2", "4")

obj.startMine()

frequentPatterns = obj.getPeriodicFrequentPatterns()

print("Total number of Frequent Patterns:", len(frequentPatterns))

obj.storePatternsInFile("patterns")

Df = obj.getPatternsInDataFrame()

memUSS = obj.getMemoryUSS()

print("Total Memory in USS:", memUSS)

memRSS = obj.getMemoryRSS()

print("Total Memory in RSS", memRSS)

run = obj.getRuntime()

print("Total ExecutionTime in seconds:", run)
