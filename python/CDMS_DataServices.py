import cdms2, os, sys
from cdms2.selectors import Selector

def getSelector( roi ):
    sel = Selector()
    return sel

def getCDMSVariable( dataset_path, var_name, roi ): 
    ds = cdsm.open( dataset_path )  
    var = ds( var_name )
    roi_selector = getSelector( roi )
    sub_var = var( roi_selector )
    return sub_var
