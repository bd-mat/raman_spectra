#%%
import pandas as pd
import numpy as np


test = pd.read_json('2dmatpedia/db.json',lines=True)
structs = test.structure
ids = test.material_id
for i in range(0,len(structs)):
    file = open('structure_database/'+ids[i]+'.json','w')
    file.write(str(structs[i]).replace("'",'"'))
    file.close()
