PAMI.fuzzyPeriodicFrequentPattern.basic package
===============================================

Submodules
----------

PAMI.fuzzyPeriodicFrequentPattern.basic.FPFPMiner module
--------------------------------------------------------

.. automodule:: PAMI.fuzzyPeriodicFrequentPattern.basic.FPFPMiner
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 FPFPMiner.py <inputFile> <outputFile> <minSup> <maxPer> <sep>
        Example:
                  >>>  python3  FPFPMiner.py sampleTDB.txt output.txt 2 3

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.fuzzyPeriodicFrequentPattern.basic import FPFPMiner as alg

        obj =alg.FPFPMiner("input.txt",2,3)

        obj.startMine()

        periodicFrequentPatterns = obj.getPatterns()

        print("Total number of Fuzzy Periodic Frequent Patterns:", len(periodicFrequentPatterns))

        obj.savePatterns("output.txt")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  Sai Chitra.B  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

