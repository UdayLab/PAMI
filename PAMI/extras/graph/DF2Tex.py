# from PAMI.extras.graph import generateLatexFileFromDataFrame as gdf
# gdf.generateLatexCode(result)
# generateLatexFileFromDatFrame is used to convert the given dataframe into LatexFile.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.graph import DF2Tex
#
#     obj = DF2Tex.generateLatex(result, "minSup", "runtime", "Algorithm")
#
#     DF2Tex.save("outputFileName.tex", latexCode)
#
#     prints statement: Latex file saved as outputFileName
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

            obj = DF2Tex.generateLatex(result, "minSup", "runtime", "Algorithm")

            DF2Tex.save("outputFileName.tex", latexCode)

            prints statement: Latex file saved as outputFileName

    """
    @staticmethod
    def generateLatex(result: pd.DataFrame, xColumn, yColumn, algorithmColumn=None) -> str:
        latexCode = ""
        titles = [xColumn, yColumn]
        legendary = pd.unique(result.iloc[:, 0].values.ravel())
        xaxisValues = result[xColumn].values
        yaxisValues = result[yColumn].values
        if algorithmColumn == None:
            algo = result.iloc[:, 0].values
        else:
            algo = result[algorithmColumn].values
        xLabel = titles[0]
        latexCode += DF2Tex.print(xaxisValues, yaxisValues, xLabel, algo, legendary, titles)
        return latexCode

    @staticmethod
    def print(xaxisValues, yaxisValues, xLabel, algo, legendary, title):
        color = ['red', 'blue', 'green', 'black', 'yellow']
        latexCode = ""
        latexCode += "\\begin{axis}[\n\txlabel={\\Huge{" + xLabel + "}},"
        latexCode += "\n\tylabel={\\Huge{" + title[1] + "}},"
        latexCode += "\n\txmin=" + str(min(xaxisValues)) + ", xmax=" + str(max(xaxisValues)) + ",]\n"
        for num in range(0, len(legendary)):
            latexCode += "\\addplot+  [" + color[num] + "]\n\tcoordinates {\n"
            for num2 in range(0, len(xaxisValues)):
                if (legendary[num] == algo[num2]):
                    latexCode += "(" + str(xaxisValues[num2]) + "," + str(yaxisValues[num2]) + ")\n"
            latexCode += "\t};   \\addlegendentry{" + legendary[num] + "}\n"
            if (num + 1 == len(legendary)):
                latexCode += "\\end{axis}"
        return latexCode

    @staticmethod
    def save(outputFileName, latexCode):
        with open(outputFileName, "w") as latexwriter:
            latexwriter.write(latexCode)
        print(f"Latex file saved as {outputFileName}")


# Example usage
result = pd.DataFrame()
# generateLatex function as four parameters dataFrame, xColumn-name, yColumn-name,
# algorithmColumn-name is optional
latexCode = DF2Tex.generateLatex(result, "minSup", "runtime", "algorithmColumn")
# save function as two parameters outputFile-name and latexCode
DF2Tex.save("outputFileName.tex", latexCode)


