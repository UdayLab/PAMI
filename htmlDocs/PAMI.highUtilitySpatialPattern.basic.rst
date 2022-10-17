PAMI.highUtilitySpatialPattern.basic package
============================================

Submodules
----------

PAMI.highUtilitySpatialPattern.basic.HDSHUIM module
---------------------------------------------------

.. automodule:: PAMI.highUtilitySpatialPattern.basic.HDSHUIM
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 HDSHUIM.py <inputFile> <outputFile> <Neighbours> <minUtil>
        Example:
                  >>>  python3 HDSHUIM.py sampleTDB.txt output.txt sampleN.txt 35



**Importing this algorithm into a python program**

.. code-block:: python

         from PAMI.highUtilityFrequentSpatialPattern.basic import HDSHUIM as alg

         obj=alg.HDSHUIM("input.txt","Neighbours.txt",35)

         obj.startMine()

         Patterns = obj.getPatterns()

         print("Total number of Spatial High-Utility Patterns:", len(Patterns))

         obj.save("output")

         memUSS = obj.getMemoryUSS()

         print("Total Memory in USS:", memUSS)

         memRSS = obj.getMemoryRSS()

         print("Total Memory in RSS", memRSS)

         run = obj.getRuntime()

         print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  Sai Chitra.B  under the supervision of Professor Rage Uday Kiran.


PAMI.highUtilitySpatialPattern.basic.SHUIM module
-------------------------------------------------

.. automodule:: PAMI.highUtilitySpatialPattern.basic.SHUIM
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 HDSHUIM.py <inputFile> <outputFile> <Neighbours> <minUtil> <separator>
        Example:
                  >>>  python3 HDSHUIM.py sampleTDB.txt output.txt sampleN.txt 35

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.highUtilitySpatialPattern.basic import SHUIM as alg

        obj=alg.SHUIM("input.txt","Neighbours.txt",35)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Spatial high utility Patterns:", len(frequentPatterns))

        obj.save("output")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  Pradeep Pallikila  under the supervision of Professor Rage Uday Kiran.


