PAMI.multipleMinimumSupportBasedFrequentPattern package
=======================================================

Submodules
----------

PAMI.multipleMinimumSupportBasedFrequentPattern.CFPGrowth module
----------------------------------------------------------------

.. automodule:: PAMI.multipleMinimumSupportBasedFrequentPattern.CFPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 CFPGrowth.py sampleTDB.txt output.txt sampleN.txt MIS
                  
        Example:
                  >>>   python3 CFPGrowth.py sampleDB.txt patterns.txt MISFile.txt ','

      


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.multipleMinimumSupportBasedFrequentPattern.basic import CFPGrowth as alg

        obj = alg.CFPGrowth(iFile, mIS)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternInDataFrame()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.

-------------------------------


PAMI.multipleMinimumSupportBasedFrequentPattern.CFPGrowthPlus module
--------------------------------------------------------------------

.. automodule:: PAMI.multipleMinimumSupportBasedFrequentPattern.CFPGrowthPlus
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>   python3 CFPGrowthPlus.py <inputFile> <outputFile> <minSup>
        Example:
                  >>>   python3 CFPGrowthPlus.py sampleTDB.txt output.txt sampleN.txt MIS ',' (it will consider "," as a separator)

       


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.multipleMinimumSupportBasedFrequentPattern.basic import CFPGrowthPlus as alg

        obj = alg.CFPGrowthPlus(iFile, mIS)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternInDataFrame()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.


