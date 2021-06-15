import HDSHUIM as alg

obj=alg.SHDSHUIs("input.txt","nighbours.txt",35)

obj.startMine()

frequentPatterns = obj.getUtilityPatterns()

print("Total number of Spatial Frequent Patterns:", len(frequentPatterns))

obj.storePatternsInFile("output")

memUSS = obj.getMemoryUSS()

print("Total Memory in USS:", memUSS)

memRSS = obj.getMemoryRSS()

print("Total Memory in RSS", memRSS)

run = obj.getRuntime()

print("Total ExecutionTime in seconds:", run)