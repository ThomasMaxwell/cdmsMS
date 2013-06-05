import cdms2, os, sys
from cdms2.selectors import Selector

def getSelector( roi ):
    sel = Selector()
    return sel

def getCDMSVariable0( arg0, dataset_path, var_name, roi ): 
    ds = cdms2.open( dataset_path )  
    var = ds( var_name )
    roi_selector = getSelector( roi )
    sub_var = var( roi_selector )
    return sub_var

def getCDMSVariable( arg0, dataset_path, var_name, roi ): 
    f = open( os.path.expanduser( '~tpmaxwel/.irods/CDMSVariableTest.log' ), 'w')
    try:
        f.write( " { " ); f.flush()
        f.write( " %s  ", str( dataset_path ) ); f.flush()
        f.write( " %s  ", str( var_name ) ); f.flush()
        f.write( " %s  ", str( roi ) ); f.flush()
    except :
        f.write( " Exception: %s ", str(err) ); f.flush()
    f.write( " } " ); f.flush()
    f.close()
    return "None"


def getCDMSVariable2( *args ): 
    return "None"
