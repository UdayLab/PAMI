# DF2Tex is used to convert the given dataframe into LatexFile.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.graph import DF2Tex
#
#     obj = DF2Tex()
#
#     obj.generateLatexCode(result, "minSup", "runtime", "algorithmColumn")
#
#     obj.print()
#
#     obj.save("outputFile.tex")
#




__copyright__ = """
Copyright (C)  2021 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import pandas as pd


class DF2Tex:
    """
    :Description:  generateLatexFileFromDatFrame is used to convert the given dataframe into LatexFile.

    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

            from PAMI.extras.graph import DF2Tex

            obj = DF2Tex()

            obj.generateLatexCode(result, "minSup", "runtime", "algorithmColumn")

            obj.print()

            obj.save("outputFile.tex")

    """

    def generateLatexCode(self, result: pd.DataFrame, xColumn, yColumn, algorithmColumn=None) -> None:
        titles = [xColumn, yColumn]
        legendary = pd.unique(result.iloc[:, 0].values.ravel())
        xaxisValues = result[xColumn].values
        yaxisValues = result[yColumn].values
        if algorithmColumn is None:
            algo = result.iloc[:, 0].values
        else:
            algo = result[algorithmColumn].values
        xLabel = titles[0]
        color = ['red', 'blue', 'green', 'black', 'yellow']
        latexCode = ""
        latexCode += "\\begin{axis}[\n\txlabel={\\Huge{" + xLabel + "}},"
        latexCode += "\n\tylabel={\\Huge{" + titles[1] + "}},"
        latexCode += "\n\txmin=" + str(min(xaxisValues)) + ", xmax=" + str(max(xaxisValues)) + ",]\n"
        for num in range(0, len(legendary)):
            latexCode += "\\addplot+  [" + color[num] + "]\n\tcoordinates {\n"
            for num2 in range(0, len(xaxisValues)):
                if legendary[num] == algo[num2]:
                    latexCode += "(" + str(xaxisValues[num2]) + "," + str(yaxisValues[num2]) + ")\n"
            latexCode += "\t};   \\addlegendentry{" + legendary[num] + "}\n"
            if num + 1 == len(legendary):
                latexCode += "\\end{axis}"
        DF2Tex.latexCode = latexCode

    def print(self):
        print(DF2Tex.latexCode)

    def save(outputFileName):
        with open(outputFileName, "w") as latexwriter:
            latexwriter.write(DF2Tex.latexCode)
        print(f"Latex file saved as {outputFileName}")


# Example usage
result = pd.DataFrame()
obj = DF2Tex()
# generateLatexCode function as four parameters dataFrame, xColumn-name, yColumn-name,
# algorithmColumn-name is optional
obj.generateLatexCode(result, "minSup", "runtime", "algorithmColumn")
# printLatexCode function prints the output of latex file
obj.print()
# save function gives the outputFile
obj.save("outputFile.tex")

