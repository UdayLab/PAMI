import GPFgrowth as Ap

apri = Ap.GPFgrowth("sampleTDB.txt", 2, 5, 0.4)

apri.startMine()

frequentPatterns = apri.getPartialPeriodicPatterns()

print("Total number of Frequent Patterns:", len(frequentPatterns))

apri.storePatternsInFile("patterns.txt")

# Df = apri.getPatternInDf()

memUSS = apri.getMemoryUSS()

print("Total Memory in USS:", memUSS)

memRSS = apri.getMemoryRSS()

print("Total Memory in RSS", memRSS)

run = apri.getRuntime()

print("Total ExecutionTime in seconds:", run)