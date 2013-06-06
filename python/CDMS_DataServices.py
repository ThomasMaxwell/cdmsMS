import cdms2, os, sys
from cdms2.selectors import Selector
from cdms2.selectors import latitudeslice, longitudeslice, levelslice, timeslice

def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def parseSelectionRange( range_str, **args ):
    isTime = args.get('time',False)
    isSlice = args.get('slice',False)
    str_range_vals = range_str.split(',')
    converted_range_vals = str_range_vals
    if isSlice:       converted_range_vals = [ (   int(val) if is_float(val) else val) for val in str_range_vals ]
    elif not isTime:  converted_range_vals = [ ( float(val) if is_float(val) else val) for val in str_range_vals ]
    return tuple( converted_range_vals )
    

def getSelector( roi ):
    sel = Selector()
    for sel_comp in roi.split(';'):
        if sel_comp.strip():
            sel_comp_items = sel_comp.split('=')
            sel_name = sel_comp_items[0].lower()
            sel_range = sel_comp_items[1]
            if sel_name.startswith('lat'):
                if sel_name.endswith('slice'):  sel.refine( latitudeslice(*parseSelectionRange( sel_range, slice=True )) )
                else:                           sel.refine( lat=parseSelectionRange( sel_range ) )
            elif sel_name.startswith('lon'):
                if sel_name.endswith('slice'):  sel.refine( longitudeslice(*parseSelectionRange( sel_range, slice=True )) )
                else:                           sel.refine( lon=parseSelectionRange( sel_range ) )
            elif sel_name.startswith('lev'):
                if sel_name.endswith('slice'):  sel.refine( levelslice(*parseSelectionRange( sel_range, slice=True )) )
                else:                           sel.refine( lev=parseSelectionRange( sel_range ) )
            elif sel_name.startswith('time'):
                if sel_name.endswith('slice'):  sel.refine( timeslice(*parseSelectionRange( sel_range, slice=True )) )
                else:                           sel.refine( time=parseSelectionRange( sel_range, time=True ) )
    return sel


def getCDMSVariable( dataset_path, var_name, roi ): 
    ds = cdms2.open( dataset_path )  
    roi_selector = getSelector( roi )
    sub_var = ds( var_name, roi_selector )
    return sub_var

def getCDMSVariableTest( arg0, dataset_path, var_name, roi ): 
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
