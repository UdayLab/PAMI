import scpgrowth as alg

obj = alg.scpgrowth("sampleTDB.txt","sampleN.txt",3,0.2)

obj.startMine()

frequentPatterns = obj.getFrequentPatterns()

print("Total number of Frequent Patterns:", len(frequentPatterns))

obj.storePatternsInFile("output.txt")

Df = obj.getPatternsInDF()

memUSS = obj.getMemoryUSS()

print("Total Memory in USS:", memUSS)

memRSS = obj.getMemoryRSS()

print("Total Memory in RSS", memRSS)

run = obj.getRuntime()

print("Total ExecutionTime in seconds:", run)
