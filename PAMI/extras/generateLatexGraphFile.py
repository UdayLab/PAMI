import pandas as pd


def generateLatexCode(result):
    latexwriter = open("Latexfile.tex", "w")
    latexwriter.write("")
    plots = int(input("How many lines do you want to plot: "))
    x_label = input("Enter xlable: ")
    y_label = input("Enter ylable: ")
    x_min = int(input("Enter xmin: "))
    x_max = int(input("Enter x_max: "))
    axis_cs = input("Enter axis cs separated by comma: ")
    axis_cs = axis_cs.split(",")
    latexwriter = open("Latexfile.tex", "a")
    latexwriter.write("\\begin{axis}[\n\txlabel={\\Huge{" + x_label + "}},")
    latexwriter.write("\n\tylabel={\\Huge{" + y_label + "}},")
    latexwriter.write("\n\txmin=" + str(x_min) + ", xmax=" + str(x_max) + ",")
    latexwriter.write("\n\tlegend style={at={(axis cs: " + str(axis_cs[0]) + "," + str(axis_cs[1]) + ")}},\n]")
    for i in range(0, plots):
        legendary = input("Enter legendary title for plot: ")
        print("Enter the color for " + legendary + ": ", end="")
        color = input()
        algo_name = input("Plot results for which algorithm? (Please enter algorithm name):")
        x_axis = input("Which column will become x axis data from dataframe: ")
        y_axis = input("Which column will become y axis data from dataframe: ")
        #xaxis = result[x_axis].values.tolist()
        #yaxis = result[y_axis].values.tolist()
        df2 = result[['minSup','runtime']] [result['algorithm'] == algo_name]
        print(df2)
        algorithm = df2.values.tolist()
        print(algorithm)

        latexwriter.write("\n\\addplot+  [" + color + "]\n\tcoordinates {\n")

        for i in range(0, len(algorithm)):
            latexwriter.write("(" + str(algorithm[i][0]) + ","+ str(algorithm[i][1]) + ")\n")
        latexwriter.write("\t};   \\addlegendentry{" + legendary + "}\n")
    latexwriter.write("\\end{axis}")
    print("Latex file generated successfully")

if __name__ == "__main__":


    data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
            'Age': [27, 24, 22, 32],
            'Address': [0, 1, 2, 3],
            'Qualification': [8, 9, 10, 11]}
    data1 = pd.DataFrame(data)
    print(data1)
    #print(data1['Name'].values.tolist())
    generateLatexCode(data1)
