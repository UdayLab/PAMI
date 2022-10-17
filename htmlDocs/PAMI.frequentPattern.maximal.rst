PAMI.frequentPattern.maximal package
====================================

Submodules
----------

PAMI.frequentPattern.maximal.MaxFPGrowth module
-----------------------------------------------

.. automodule:: PAMI.frequentPattern.maximal.MaxFPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>> python3 MaxFPGrowth.py <inputFile> <outputFile> <minSup>

        Example:
                  >>> python3 MaxFPGrowth.py sampleDB.txt patterns.txt 0.3

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.frequentPattern.maximal import MaxFPGrowth as alg

        obj = alg.MaxFPGrowth("../basic/sampleTDB.txt", "2")

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns("patterns")

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)


**Credits:**

            The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.
