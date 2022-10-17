PAMI.uncertainPeriodicFrequentPattern.basic package
===================================================

Submodules
----------

PAMI.uncertainPeriodicFrequentPattern.basic.UPFPGrowth module
-------------------------------------------------------------

.. automodule:: PAMI.uncertainPeriodicFrequentPattern.basic.UPFPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 UPFPGrowth.py <inputFile> <outputFile> <minSup> <maxPer>
        Example:
                  >>>  python3 UPFPGrowth.py sampleTDB.txt patterns.txt 0.3 4

                 .. note:: minSup and maxPer will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.uncertainPeriodicFrequentPattern import UPFPGrowth as alg

        obj = alg.UPFPGrowth(iFile, minSup, maxPer)

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

