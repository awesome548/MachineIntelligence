def fix(path, subject_num):
    for i in range(subject_num):
        f = open(path + "person_" + str(i).zfill(3) + ".csv")
        f_o = open(path + "person_" + str(i).zfill(3) + "_out.csv", "w")

        line = f.readline()
        while line:
            f_o.writelines([line[:-7], "\n"])
            line = f.readline()

        f.close()
        f_o.close()
    return