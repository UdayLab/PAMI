import CFFI as ap

ap =ap.CFFI("input.txt",3,0.2)

ap.startMine()

frequentPatterns = ap.getFrequentPatterns()

print("Total number of Corelated Fuzzy Frequent Patterns:", len(frequentPatterns))

ap.storePatternsInFile("outp")

memUSS = ap.getMemoryUSS()

print("Total Memory in USS:", memUSS)

memRSS = ap.getMemoryRSS()

print("Total Memory in RSS", memRSS)

run = ap.getRuntime

print("Total ExecutionTime in seconds:", run)