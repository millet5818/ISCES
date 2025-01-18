import numpy as np
import pandas as pd
from scipy.stats import pearsonr

data=pd.read_csv("D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\Deformation\Co_年变形_indicator1.csv")

Point_Name=data.columns[1:74]

CE_Type=['R95P', 'R95D', 'R95C', 'R95DM', 'CEF', 'CEA',  'CED', 'T95D', 'T95Max_L', 'T95End_T', 'T95Start_T', 'T95Events']

CES_Type=['R95P', 'R95D', 'R95C', 'R95DM', 'CESF','CESA',  'CESD','T95D', 'T95Max_L', 'T95End_T', 'T95Start_T', 'T95Events']

CET_Type=['R95P', 'R95D', 'R95C', 'R95DM', 'CETF', 'CETA',  'CETD','T95D', 'T95Max_L', 'T95End_T', 'T95Start_T', 'T95Events']

CETS_Type=['R95P', 'R95D', 'R95C', 'R95DM', 'CETSF',  'CETSA','CETSD', 'T95D', 'T95Max_L', 'T95End_T', 'T95Start_T', 'T95Events']

Corre_Array=np.zeros((len(Point_Name),len(CETS_Type)))
Corre_Array_p=np.zeros((len(Point_Name),len(CETS_Type)))

for j in range(len(Point_Name)):
    for i in range(len(CETS_Type)):
        correlation, p_value = pearsonr(data[CETS_Type[i]] ,data[Point_Name[j]])
        Corre_Array[j,i]=correlation
        Corre_Array_p[j,i]=p_value

df=pd.DataFrame(Corre_Array,columns=CETS_Type)
df.to_csv("D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\Deformation\CETS_Corref_年变形_indicator.csv",index=False)

df_v=pd.DataFrame(Corre_Array_p,columns=CETS_Type)
df_v.to_csv("D:\zhaozheng\projects\全球风险计算\SBAS_INSAR\ZZ\Deformation\CETS_Corre_Value_年变形_indicator.csv",index=False)
print(111)
