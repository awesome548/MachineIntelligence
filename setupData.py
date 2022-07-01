f = open("./Dataset/myData/kikuzo0701.csv")
f_o = open("./Dataset/myData/kikuzo0701_out.csv", "w")

line = f.readline()
while line:
    f_o.writelines([line[:-7], "\n"])
    line = f.readline()

f.close()
f_o.close()