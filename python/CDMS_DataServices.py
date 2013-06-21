import cdms2, os, sys
from cdms2.selectors import Selector
from cdms2.selectors import latitudeslice, longitudeslice, levelslice, timeslice

def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def parseSelectionRange( range_obj, **args ):
    if type( range_obj ) <> type( "" ): return range_obj
    isTime = args.get('time',False)
    isSlice = args.get('slice',False)
    str_range_vals = range_obj.split(',')
    converted_range_vals = str_range_vals
    if isSlice:       converted_range_vals = [ (   int(val) if is_float(val) else val) for val in str_range_vals ]
    elif not isTime:  converted_range_vals = [ ( float(val) if is_float(val) else val) for val in str_range_vals ]
    return tuple( converted_range_vals )
  
#   def parse_UVCDAT_roi_syntax( roi ):
#     roi = roi.strip("{}")
#     inRangeDecl = False
#     roi_map_items, roi_map = [], []
#     start_index = 0
#     for sIndex in range( len(roi) ):
#         c = roi[ sIndex ]
#         if c == '(': inRangeDecl = True
#         elif c == ')': inRangeDecl = False
#         elif c == ',' and not inRangeDecl:
#             roi_map_items.append( roi[start_index:sIndex] ) 
#             start_index = sIndex + 1
#     roi_map_items.append( roi[start_index:-1] ) 
#     for roi_map_item in roi_map_items:
#         roi_map.append( roi_map_item.strip(" ,()").split(":") )
#     return roi_map
  

def refineSelector( sel, sel_name, sel_range ):
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

def getSelector( roi ):
    sel = Selector()
    if roi[0] == '{':
        roi_map = eval( roi )
        for items in roi_map.items():
            sel_name = items[0].lower()
            sel_range = items[1]
            refineSelector( sel, sel_name, sel_range )               
    else:
        for sel_comp in roi.split(';'):
            if sel_comp.strip():
                sel_comp_items = sel_comp.split('=')
                sel_name = sel_comp_items[0].lower()
                sel_range = sel_comp_items[1]
                refineSelector( sel, sel_name, sel_range )               
    return sel

def getCDMSVariable( user_name, dataset_path, var_name, roi ): 
    ds = cdms2.open( dataset_path )  
    roi_selector = getSelector( roi )
    sub_var = ds( var_name, roi_selector )
    return sub_var

def transferCDMSVariable( user_name, dataset_path, var_name, roi ): 
    ds = cdms2.open( dataset_path )  
    roi_selector = getSelector( roi.strip() )
    sub_var = ds( var_name, roi_selector )
    dspath_parts = os.path.split( dataset_path )
    subdir = dspath_parts[0]
    ds_dir = os.path.splitext( dspath_parts[1] )[0]
    file_name = '_'.join( [ user_name, subdir, ds_dir, var_name+'.nc' ] ).replace( '/', '_' )
    new_path =  '/tmp/' + file_name
    new_ds = cdms2.open( new_path, 'w' )  
    new_ds.write( sub_var )
    new_ds.close()
    return new_path

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
