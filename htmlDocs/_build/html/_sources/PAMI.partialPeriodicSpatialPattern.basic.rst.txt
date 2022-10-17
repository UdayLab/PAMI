PAMI.partialPeriodicSpatialPattern.basic package
================================================

Submodules
----------

PAMI.partialPeriodicSpatialPattern.basic.STEclat module
-------------------------------------------------------

.. automodule:: PAMI.partialPeriodicSpatialPattern.basic.STEclat
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 STEclat.py <inputFile> <outputFile> <neighbourFile>  <minPS>  <maxIAT>
        Example:
                  >>>  python3 STEclat.py sampleTDB.txt output.txt sampleN.txt 0.2 0.5

        .. note:: maxIAT & minPS will be considered in support count or frequency


**Importing this algorithm into a python program**

.. code-block:: python

        import PAMI.partialPeriodicSpatialPattern.basic.STEclat as alg

        obj = alg.STEclat("sampleTDB.txt", "sampleN.txt", 3, 4)

        obj.startMine()

        partialPeriodicSpatialPatterns = obj.getPatterns()

        print("Total number of Periodic Spatial Frequent Patterns:", len(partialPeriodicSpatialPatterns))

        obj.savePatterns("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.

