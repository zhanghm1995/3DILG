import os.path
rootdir = "/mntnfs/cui_data4/yanchengwang/Poisson_sample/2048x16/"
i=0;
for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        # print("parent is: " + parent)
        # print("filename is: " + filename)
        print(os.path.join(parent, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
        newName=filename.replace('.off', '.xyz')
        print(newName)
        os.rename(os.path.join(parent, filename), os.path.join(parent, newName))
        i=i+1