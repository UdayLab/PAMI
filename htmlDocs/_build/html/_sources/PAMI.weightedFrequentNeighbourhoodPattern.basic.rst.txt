PAMI.weightedFrequentNeighbourhoodPattern.basic package
=======================================================

Submodules
----------

PAMI.weightedFrequentNeighbourhoodPattern.basic.SWFPGrowth module
-----------------------------------------------------------------

.. automodule:: PAMI.weightedFrequentNeighbourhoodPattern.basic.SWFPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 SWFPGrowth.py <inputFile> <weightFile> <outputFile> <minSup> <minWeight>
        Example:
                  >>>  python3 SWFPGrowth.py sampleDB.txt weightFile.txt patterns.txt 10  2

                 .. note:: minSup will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.weightFrequentNeighbourhoodPattern.basic import SWFPGrowth as alg

        obj = alg.SWFPGrowth(iFile, wFile, nFile, minSup, minWeight, seperator)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getmemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)
**Credits:**

         The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



