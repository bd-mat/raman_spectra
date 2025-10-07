#%%
import pandas as pd
import numpy as np


test = pd.read_json('2dmatpedia/db.json',lines=True)
structs = test.structure
ids = test.material_id
for line in structs:
    file = open('structure_database/'+str())
    print(line)