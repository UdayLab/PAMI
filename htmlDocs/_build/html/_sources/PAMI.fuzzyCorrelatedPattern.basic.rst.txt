PAMI.fuzzyCorrelatedPattern.basic package
=========================================

Submodules
----------

PAMI.fuzzyCorrelatedPattern.basic.FCPGrowth module
--------------------------------------------------

.. automodule:: PAMI.fuzzyCorrelatedPattern.basic.FCPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 FCPGrowth.py <inputFile> <outputFile> <minSup> <minAllConf> <sep>

        Example:
                  >>> python3 FCPGrowth.py sampleTDB.txt output.txt 2 0.2

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.fuzzyCorrelatedPattern.basic import FCPGrowth as alg

        obj = alg.FCPGrowth("input.txt",2,0.4)

        obj.startTimeMine()

        correlatedFuzzyFrequentPatterns = obj.getPatterns()

        print("Total number of Correlated Fuzzy Frequent Patterns:", len(correlatedFuzzyFrequentPatterns))

        obj.savePatterns("output")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime

        print("Total ExecutionTime in seconds:", run)


**Credits:**

         The complete program was written by  Sai Chitra.B  under the supervision of Professor Rage Uday Kiran.