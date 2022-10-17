PAMI.stablePeriodicFrequentPattern.basic package
================================================

Submodules
----------

PAMI.stablePeriodicFrequentPattern.basic.SPPECLAT module
--------------------------------------------------------

.. automodule:: PAMI.stablePeriodicFrequentPattern.basic.SPPECLAT
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>   python3 SPPECLAT.py <inputFile> <outputFile> <minSup> <maxPer> <maxLa>

        Example:
                  >>>    python3 SPPECLAT.py sampleDB.txt patterns.txt 10.0 4.0 2.0

        .. note:: constraints will be considered in percentage of database transactions

**Importing this algorithm into a python program**

.. code-block:: python

                from PAMI.stablePeriodicFrequentPattern.basic import SPPECLAT as alg

                obj = alg.PFPECLAT("../basic/sampleTDB.txt", 5, 3, 3)

                obj.startMine()

                Patterns = obj.getPatterns()

                print("Total number of Stable Periodic Frequent Patterns:", len(Patterns))

                obj.save("patterns")

                Df = obj.getPatternsAsDataFrame()

                memUSS = obj.getMemoryUSS()

                print("Total Memory in USS:", memUSS)

                memRSS = obj.getMemoryRSS()

                print("Total Memory in RSS", memRSS)

                run = obj.getRuntime()

                print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  P.Likhitha under the supervision of Professor Rage Uday Kiran.


PAMI.stablePeriodicFrequentPattern.basic.SPPGrowth module
---------------------------------------------------------

.. automodule:: PAMI.stablePeriodicFrequentPattern.basic.SPPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>   python3 SPPGrowth.py <inputFile> <outputFile> <minSup> <maxPer> <maxLa>
        Example:
                  >>>  python3 SPPGrowth.py sampleTDB.txt patterns.txt 0.3 0.4 0.3

        .. note:: constraints will be considered in percentage of database transactions

**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.stablePeriodicFrequentPattern.basic import SPPGrowth as alg

                obj = alg.SPPGrowth(iFile, minSup, maxPer, maxLa)

                obj.startMine()

                Patterns = obj.getPatterns()

                print("Total number of Stable Periodic Frequent Patterns:", len(Patterns))

                obj.save(oFile)

                Df = obj.getPatternsAsDataFrame()

                memUSS = obj.getMemoryUSS()

                print("Total Memory in USS:", memUSS)

                memRSS = obj.getMemoryRSS()

                print("Total Memory in RSS", memRSS)

                run = obj.getRuntime()

                print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  P.Likhitha under the supervision of Professor Rage Uday Kiran.


PAMI.stablePeriodicFrequentPattern.basic.SPPGrowthDump module
-------------------------------------------------------------

.. automodule:: PAMI.stablePeriodicFrequentPattern.basic.SPPGrowthDump
   :members:
   :undoc-members:
   :show-inheritance:

