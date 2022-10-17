PAMI.frequentSpatialPattern.basic package
=========================================

Submodules
----------

PAMI.frequentSpatialPattern.basic.FSPGrowth module
--------------------------------------------------

.. automodule:: PAMI.frequentSpatialPattern.basic.FSPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 FSPGrowth.py <inputFile> <outputFile> <neighbourFile> <minSup>

        Example:
                  >>>  python3 FSPGrowth.py sampleTDB.txt output.txt sampleN.txt 0.5

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.frequentSpatialPattern.basic import FSPGrowth as alg

        obj = alg.FSPGrowth("sampleTDB.txt", "sampleN.txt", 5)

        obj.startMine()

        spatialFrequentPatterns = obj.getPatterns()

        print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))

        obj.savePatterns("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)


**Credits:**

         The complete program was written by  Yudai Masu under the supervision of Professor Rage Uday Kiran.



PAMI.frequentSpatialPattern.basic.SpatialECLAT module
-----------------------------------------------------

.. automodule:: PAMI.frequentSpatialPattern.basic.SpatialECLAT
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 SpatialECLAT.py <inputFile> <outputFile> <neighbourFile> <minSup>

        Example:
                  >>>  python3 SpatialECLAT.py sampleTDB.txt output.txt sampleN.txt 0.5

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.frequentSpatialPattern.basic import SpatialECLAT as alg

        obj = alg.SpatialECLAT("sampleTDB.txt", "sampleN.txt", 5)

        obj.startMine()

        spatialFrequentPatterns = obj.getPatterns()

        print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))

        obj.savePatterns("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)


**Credits:**

         The complete program was written by  Sai Chitra.B  under the supervision of Professor Rage Uday Kiran.

