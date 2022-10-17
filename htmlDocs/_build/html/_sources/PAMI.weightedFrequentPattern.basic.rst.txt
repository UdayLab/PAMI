PAMI.weightedFrequentPattern.basic package
==========================================

Submodules
----------

PAMI.weightedFrequentPattern.basic.WFIM module
----------------------------------------------

.. automodule:: PAMI.weightedFrequentPattern.basic.WFIM
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>>  python3 WFIM.py <inputFile> <weightFile> <outputFile> <minSup> <minWeight>
        Example:
                  >>>  python3 WFIM.py sampleDB.txt weightSample.txt patterns.txt 10.0 3.4

                 .. note:: minSup and maxPer will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.weightFrequentPattern.basic import WFIM as alg

        obj = alg.WFIM(iFile, wFile, minSup, minWeight)

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

