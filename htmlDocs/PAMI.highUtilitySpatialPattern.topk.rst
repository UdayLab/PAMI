PAMI.highUtilitySpatialPattern.topk package
===========================================

Submodules
----------

PAMI.highUtilitySpatialPattern.topk.TKSHUIM module
--------------------------------------------------

.. automodule:: PAMI.highUtilitySpatialPattern.topk.TKSHUIM
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 TKSHUIM.py <inputFile> <outputFile> <Neighbours> <k> <sep>
        Example:
                  >>>  python3 TKSHUIM.py sampleTDB.txt output.txt sampleN.txt 35

       


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.highUtilitySpatialPattern.topk import TKSHUIM as alg

        obj=alg.TKSHUIM("input.txt","Neighbours.txt",35)

        obj.startMine()

        Patterns = obj.getPatterns()

        obj.savePatterns("output")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by   Pradeep Pallikila under the supervision of Professor Rage Uday Kiran.


