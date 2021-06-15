import cpgrowthpp as cfp

CFPG = cfp.cpgrowthpp("input.txt","4",0.3)

CFPG.startMine()

frequentPatterns = CFPG.getFrequentPatterns()

print("Total number of Frequent Patterns:", len(frequentPatterns))

CFPG.storePatternsInFile("outFile")

memUSS = CFPG.getMemoryUSS()

print("Total Memory in USS:", memUSS)

memRSS = CFPG.getMemoryRSS()

print("Total Memory in RSS", memRSS)

run = CFPG.getRuntime()

print("Total ExecutionTime in seconds:", run)

"""
Credits:
-------
	The complete program was written by Sai Chitra.B under the supervision of Professor Rage Uday Kiran.
	The complete verification and documentation is done by Penugonda Ravikumar.

"""