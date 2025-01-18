import xarray as xr
import numpy as np
import pandas as pd


Path=['CE','CES','CET','CETS']
file=[f'E:\CE_DATA\Data_Processing\Average_Mode\\{i}\\CMIP6\\{i}_Start_585.nc' for i in Path]
for i in range(len(file)):
    data=xr.open_dataset(file[i])
    Y_M_D=data[f'{Path[i]}_Start'].sel(quantile=0.7).mean(dim=['lat','lon'])
    Y_M_D.to_pandas().to_csv(f"E:\CE_DATA\Data_Processing\Average_Mode\\{Path[i]}_585_st.csv")
    print(1)
