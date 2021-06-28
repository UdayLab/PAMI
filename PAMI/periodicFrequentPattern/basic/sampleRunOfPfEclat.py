from PAMI.periodicFrequentPattern.basic import PFEclat as alg

obj = alg.PFEclat("sampleTDB.txt", "2", "4")
obj.startMine()

frequentPatterns = obj.getPeriodicFrequentPatterns()

print("Total number of Frequent Patterns:", len(frequentPatterns))

obj.storePatternsInFile("patterns")

# Df = apri.getPatternInDf()

memUSS = obj.getMemoryUSS()

print("Total Memory in USS:", memUSS)

memRSS = obj.getMemoryRSS()

print("Total Memory in RSS", memRSS)

run = obj.getRuntime()

print("Total ExecutionTime in seconds:", run)


