import shutil
import json
import os

directory = 'data'

for filename in os.listdir(directory):
    folderOk = False
    for finalName in os.listdir(os.path.join(directory, filename)):
        if finalName == 'ExpertPlusStandard.dat' or finalName == 'ExpertPlus.dat':
            #print(os.path.join(os.path.join(directory, filename), finalName))
            folderOk = True
            file = open(os.path.join(os.path.join(directory, filename), finalName), 'r', encoding="utf-8")
            string = file.read()
            #print(string[:1000])
            notes = json.loads(string)['_notes']
            #print(os.path.join(os.path.join(directory, filename), finalName))
            if (len(json.loads(string)['_obstacles']) != 0):
                folderOk = False
            for x in range(0, len(notes)):
                if (notes[x]['_type'] > 3 or notes[x]['_cutDirection'] > 8 or notes[x]['_lineIndex'] > 3 or notes[x]['_lineLayer'] > 2):
                    folderOk = False
                    continue
            file.close()
    if not folderOk:
        shutil.rmtree(os.path.join(directory, filename))
