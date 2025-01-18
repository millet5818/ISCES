import xarray as xr


class Files_Read:
    def __init__(self,file_list):
        self.file_list=file_list
    def Read_File_Array(self):
        with xr.open_mfdataset(self.file_list,concat_dim="time", combine="nested",data_vars='minimal', coords='minimal', compat='override') as file_list:
            Variable =[j for j in file_list.variables.keys()]
            Time_Array=file_list.time.dt.year
            return Variable,Time_Array
