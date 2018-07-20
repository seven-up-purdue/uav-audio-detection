from dataProcess import Sound

def main():
    load = Sound("../../soundData/0720A/load/", "wav")
    load.load()
    unload = Sound("../../soundData/0720A/unload/", "wav")
    unload.load()


    return


if __name__ == "__main__":
    main()






"""
data1: get from Wright state univ
- load: ../../soundData/0701A/load/
- unload: ../../soundData/0701A/unload/

data2: make in Armory (Master student)
- load: ../../soundData/0720A/load/
- unload: ../../soundData/0720A/unload/

data3: make in Armory (Notebook)
- load: ../../soundData/0720B/load/
- unload: ../../soundData/0720B/unload/ 

data4: make in Armory (Raspberry)
- load: ../../soundData/0720C/load/
- unload: ../../soundData/0720C/unload/

"""