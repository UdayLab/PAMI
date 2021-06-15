from PAMI.frequentSpatialPattern.basic import SpatialEclat as alg

SEclat = alg.SpatialEclat("input.txt", "neighbours.txt", 5)

SEclat.startMine()

frequentPatterns = SEclat.getFrequentPatterns()

print("Total number of Spatial Frequent Patterns:", len(frequentPatterns))

SEclat.storePatternsInFile("outFile")

memUSS = SEclat.getMemoryUSS()

print("Total Memory in USS:", memUSS)

memRSS = SEclat.getMemoryRSS()

print("Total Memory in RSS", memRSS)

run = SEclat.getRuntime()

print("Total ExecutionTime in seconds:", run)
