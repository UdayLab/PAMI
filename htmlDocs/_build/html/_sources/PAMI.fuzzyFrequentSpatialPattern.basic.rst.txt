PAMI.fuzzyFrequentSpatialPattern.basic package
==============================================

Submodules
----------

PAMI.fuzzyFrequentSpatialPattern.basic.FFSPMiner module
-------------------------------------------------------

.. automodule:: PAMI.fuzzyFrequentSpatialPattern.basic.FFSPMiner
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 FFSPMiner.py <inputFile> <outputFile> <neighbours> <minSup> <sep>

        Example:
                  >>>  python3  FFSPMiner.py sampleTDB.txt output.txt sampleN.txt 3

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.fuzzyFrequentSpatialPattern import FFSPMiner as alg

        obj = alg.FFSPMiner("input.txt", "neighbours.txt", 2)

        obj.startMine()

        fuzzySpatialFrequentPatterns = obj.getPatterns()

        print("Total number of fuzzy frequent spatial patterns:", len(fuzzySpatialFrequentPatterns))

        obj.savePatterns("outputFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  Sai Chitra.B  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

