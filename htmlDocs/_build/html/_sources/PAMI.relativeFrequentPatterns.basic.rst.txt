PAMI.relativeFrequentPatterns.basic package
===========================================

Submodules
----------

PAMI.relativeFrequentPatterns.basic.RSFPGrowth module
-----------------------------------------------------

.. automodule:: PAMI.relativeFrequentPatterns.basic.RSFPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 RSFPGrowth.py <inputFile> <outputFile> <minSup> <__minRatio>
        Example:
                  >>>  python3 RSFPGrowth.py sampleDB.txt patterns.txt 0.23 0.2

        .. note:: maxPer and minPS will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.frequentPatternUsingOtherMeasures import RSFPGrowth as alg

        obj = alg.RSFPGrowth(iFile, minSup, __minRatio)

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

         The complete program was written by   Sai Chitra.B   under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

