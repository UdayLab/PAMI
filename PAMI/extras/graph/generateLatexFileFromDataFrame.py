import pandas as pd


def generateLatexCode(result):

    titles = result.columns.tolist()
    titles.remove("minSup")
    titles.remove("algorithm")
    for i in range(0, len(titles)):
        legendary = pd.unique(result[['algorithm']].values.ravel())
        color = ['red', 'blue', 'green', 'black', 'yellow']
        xaxis = result["minSup"].values.tolist()
        yaxis = result[titles[i]].values.tolist()
        algo = result["algorithm"].values.tolist()
        x_label = "minSup"
        filename = titles[i]
        latexwriter = open(filename + "Latexfile.tex", "w")
        latexwriter.write("")
        latexwriter.write("\\begin{axis}[\n\txlabel={\\Huge{" + x_label + "}},")
        latexwriter.write("\n\tylabel={\\Huge{" + titles[i] + "}},")
        latexwriter.write("\n\txmin=" + str(min(xaxis)) + ", xmax=" + str(max(xaxis)) + ",]")

        for num in range(0, len(legendary)):
            latexwriter.write("\n\\addplot+  [" + color[num] + "]\n\tcoordinates {\n")
            for num2 in range(0, len(xaxis)):
                if (legendary[num] == algo[num2]):
                    latexwriter.write("(" + str(xaxis[num2]) + "," + str(yaxis[num2]) + ")\n")
            latexwriter.write("\t};   \\addlegendentry{" + legendary[num] + "}\n")
            if (num + 1 == len(legendary)):
                latexwriter.write("\\end{axis}")
    print("Latex files generated successfully")
    #data1 = pd.DataFrame(data)
    #generateLatexCode(data1)

if __name__ == "__main__":


    #data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
            #'Age': [27, 24, 22, 32],
            #'Address': [0, 1, 2, 3],
            #'Qualification': [8, 9, 10, 11]}
    data = {'algorithm': ['FGPFPMiner','FGPFPMiner','FGPFPMiner','FGPFPMiner','FGPFPMiner','FGPFPMiner','FGPFPMiner'
        ,'Naive algorithm','Naive algorithm','Naive algorithm','Naive algorithm','Naive algorithm','Naive algorithm'
        ,'Naive algorithm', ],
            'minSup': [200,400,600,800,1000,1200,1400,200,400,600,800,1000,1200,1400],
            'patterns': [25510,5826,2305,1163,657,407,266,101938,16183,5027,2091,1044,574,335],
            'runtime': [1077.7172002792358,298.6219701766968,186.86728835105896,126.96730422973633
                ,77.39371657371521,64.73982691764832,46.879486083984375,13175.030002832413,1821.2089745998383
                ,964.6961390972137,637.1588702201843,350.71105194091797,275.9953947067261,195.6615695953369],
            'memoryRSS': [164634624,159494144,157622272,156184576,153698304,150597632,149381120,228220928,192770048
                ,185114624,182939648,178253824,176115712,171659264],
            'memoryUSS': [144310272,139104256,137232384,135794688,133300224,130195456,128978944,
                        203337728,172376064,164720640,162545664,157859840,155721728,151265280]
            }

    data1 = pd.DataFrame(data)
    #print(data1)
    #print(data1['Name'].values.tolist())
    generateLatexCode(data1)
