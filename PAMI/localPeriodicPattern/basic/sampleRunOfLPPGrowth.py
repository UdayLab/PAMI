import LPPGrowth as alg

obj = alg.LPPGrowth('sampleTDB.txt', 0.2, 0.3, 0.5)
obj.startMine()
localPeriodicPatterns = obj.getLocalPeriodicPatterns()
print(f'Total number of local periodic patterns: {len(localPeriodicPatterns)}')
obj.storePatternsInFile('patterns.txt')
Df = obj.getPatternsInDataFrame()
memUSS = obj.getMemoryUSS()
print(f'Total memory in USS: {memUSS}')
memRSS = obj.getMemoryRSS()
print(f'Total memory in RSS: {memRSS}')
runtime = obj.getRuntime()
print(f'Total execution time in seconds: {runtime}')