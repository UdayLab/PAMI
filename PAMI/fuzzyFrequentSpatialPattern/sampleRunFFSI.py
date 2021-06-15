import FFSI as alg

obj = alg.FFSI("input.txt", "neighbours.txt", 2)

obj.startMine()

frequentPatterns = obj.getFrequentPatterns()

print("Total number of Spatial Frequent Patterns:", len(frequentPatterns))

obj.storePatternsInFile("outp")

memUSS = obj.getMemoryUSS()

print("Total Memory in USS:", memUSS)

memRSS = obj.getMemoryRSS()

print("Total Memory in RSS", memRSS)

run = obj.getRuntime()

print("Total ExecutionTime in seconds:", run)
