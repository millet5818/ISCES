"""
定义一些公共组件模板
"""

class CommonVar:
    EC_Methods_List=['Relative threshold','Absolute threshold','Probability distribution']
    EC_Index_List=['Pre_Related','Tem_Related','R95P','R95D','R95I','R95F','R10mmF','R10mmD','TP90D','TP90F','TPMaxLength']
    CE_Methods_List=["TS-Compound","T-Compound","S-Compound"]
    CE_Index_List=["HPHTD1","HPHTD2","HPHTD3","HPHTD4","HPHTF1","HPHTF2","HPHTF3","HPHTF4","HPHTF5","HPHTF6","HPHTM1","HPHTM2","HPHTM3","HPHTEN1","HPHTEN2","HPHTEN3","HPHTRA1","HPHTRA2","HPHTRA3"]
    Interpolation_Method=['Bilinear','Cubic','Nearest']
    Downsacaling_Method=['Delta','Quantile Mapping','Principal Components']
