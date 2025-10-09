#%%
import pandas as pd
import numpy as np


test = pd.read_json('2dmatpedia/db.json',lines=True)
structs = test.structure
bandgaps = test.bandgap
ids = test.material_id
# writing structure files
"""
for i in range(0,len(structs)):
    file = open('structure_database/'+ids[i]+'.json','w')
    file.write(str(structs[i]).replace("'",'"'))
    file.close()
"""
# writing id_prop.csv
file = open('structure_database/id_prop.csv','w')
for i in range(0,len(bandgaps)):
    file.write(str(ids[i])+','+str(bandgaps[i])+"\n")
file.close()