PAMI.fuzzySpatialPeriodicFrequentPattern package
================================================

Submodules
----------

PAMI.fuzzySpatialPeriodicFrequentPattern.FGPFPMiner module
----------------------------------------------------------

.. automodule:: PAMI.fuzzySpatialPeriodicFrequentPattern.FGPFPMiner
   :members:
   :undoc-members:
   :show-inheritance:



**Methods to execute code on terminal**

        Format:
                  >>>  python3 FGPFPMiner.py <inputFile> <outputFile> <neighbours> <minSup> <sep>
        Example:
                  >>> python3  FGPFPMiner.py sampleTDB.txt output.txt sampleN.txt 3

        .. note:: minSup will be considered in support count or frequency

**Importing this algorithm into a python program**

.. code-block:: python


        from PAMI.fuzzyFrequentSpatialPattern import FGPFPMiner as alg

        obj = alg.FGPFPMiner("input.txt", "neighbours.txt", 2)

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

         The complete program was written by  B.Sai Chitra and Kundai Kwangwari        under the supervision of Professor Rage Uday Kiran.


