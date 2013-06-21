#include <cstdarg>
#include "rods.h"
#include "cdmsMS.h"
#include "dataObjOpr.h"
#include "uvcdatWrappers.h"
#include "ncOpen.h"
#include "ncInqId.h"
#include "ncGetVarsByType.h"
#include "nccfGetVara.h"
#include "putUtil.h"
#define BIG_STR 2000

void cdmsLog( char *formatStr, ... ) {
	char bigString[BIG_STR];
	va_list ap;
	va_start(ap, formatStr);
	vsnprintf(bigString, BIG_STR-1, formatStr, ap);
	va_end(ap);
	rodsLog(LOG_NOTICE, "<-------------------------> CDMS Message <------------------------->\n     %s  ", bigString );
}

int setDataArrayType( ncGetVarOut_t *ncGetVarOut, void* arr ) {
	int type = NETCDF_INVALID_DATA_TYPE;
	if( isString( arr ) ) {
		type = NC_CHAR;
		rstrcpy (ncGetVarOut->dataType_PI, "charDataArray_PI", NAME_LEN);
	} else if( isInteger(arr) ) {
		if( isSigned(arr) ) {
		   type = NC_INT;
		   rstrcpy (ncGetVarOut->dataType_PI, "intDataArray_PI", NAME_LEN);
		} else {
		   type = NC_UINT;
		   rstrcpy (ncGetVarOut->dataType_PI, "intDataArray_PI", NAME_LEN);
		}
	} else if( isFloat(arr) ) {
		   type = NC_FLOAT;
		   rstrcpy (ncGetVarOut->dataType_PI, "intDataArray_PI", NAME_LEN);
	} else {
		 rodsLog (LOG_ERROR, "msiGetCDMSVariable:setDataArrayType- Unsupported dataType: '%c'", getTypeDesc( arr ) );
	}
	ncGetVarOut->dataArray->type = type;
	return type;
}


int msiGetCDMSVariable( msParam_t *mspDatasetPath, msParam_t *mspVarName, msParam_t *mspRoi, msParam_t *mspResult, ruleExecInfo_t *rei)
{
	RE_TEST_MACRO( "    Calling msiGetCDMSVariable");
	cdmsLog( "Calling msiGetCDMSVariable" );

//	PyGILState_STATE gstate = PyGILState_Ensure();
//	char *strFunctionName = "getCDMSVariable";
//	char *strScriptName = "CDMS_DataServices.py";

	char *strVarName = parseMspForStr(mspVarName);
	char *strRoi = parseMspForStr(mspRoi);
	rsComm_t *rsComm = rei->rsComm;
	int err_code = 0, status;

	char* strZone = rsComm->myEnv.rodsZone;
	char* userName = rsComm->myEnv.rodsUserName;
	char* strRodsHome = rsComm->myEnv.rodsHome;
	char* strRodsEnvFile = getRodsEnvFileName();
    dataObjInp_t dataObjInp, *myDataObjInp;
    if ((rei->status = parseMspForDataObjInp ( mspDatasetPath, &dataObjInp, &myDataObjInp, 0)) < 0) {
        rodsLogAndErrorMsg (LOG_ERROR, &rsComm->rError, rei->status, "msiGetCDMSVariable: input mspDatasetPath error. status = %d", rei->status);
        return (rei->status);
    }
	dataObjInfo_t *dsetPathObjInfo = NULL;
	status = getDataObjInfo (rsComm, myDataObjInp, &dsetPathObjInfo, NULL, 1);
    if (status < 0) {
        rodsLog ( LOG_ERROR, "msiGetCDMSVariable: getDataObjInfo for %s", myDataObjInp->objPath );
        return (status);
    }
	char *strDatasetPhysicalPath = dsetPathObjInfo->filePath;

	// Call the python microservice with the parameters
	cdmsLog( " Call function getVariable( '%s', '%s', '%s' ) ", strDatasetPhysicalPath, strVarName, strRoi );
	void *arr = getVariable( userName, strDatasetPhysicalPath, strVarName, strRoi );
	if ( arr == NULL ) {
        rodsLog ( LOG_ERROR, "Error Getting Variable." );
        return (status);
	}
	ncGetVarOut_t *ncGetVarOut = (ncGetVarOut_t *) calloc (1, sizeof (ncGetVarOut_t));
	ncGetVarOut->dataArray = (dataArray_t *) calloc (1, sizeof (dataArray_t));
	int size = getSize( arr );
	int item_size = getItemSize( arr );
	if( size < 0 ) { rodsLog ( LOG_ERROR, " Error getting data size. " ); }
	cdmsLog( " Data Size = %d, Item size = %d ", size, item_size );
	ncGetVarOut->dataArray->len = size;
	int dtype = setDataArrayType( ncGetVarOut, arr );
	if( dtype == NETCDF_INVALID_DATA_TYPE ) { err_code = INVALID_OBJECT_TYPE; }
	int nbytes = sizeof (int) * size;
	ncGetVarOut->dataArray->buf = calloc (1, nbytes);
	memcpy( ncGetVarOut->dataArray->buf, getRawData( arr ), nbytes );

	rei->status = err_code;
	if (rei->status >= 0) {
		cdmsLog( " Filling MsParam, Data Length = %d, type = %s (%d) ", ncGetVarOut->dataArray->len, ncGetVarOut->dataType_PI, ncGetVarOut->dataArray->type );
		fillMsParam ( mspResult, NULL, NcGetVarOut_MS_T, &ncGetVarOut, NULL );
	} else {
		rodsLogAndErrorMsg( LOG_ERROR, &rsComm->rError, rei->status, "msiGetCDMSVariable failed, status = %d", rei->status );
	}

	cdmsLog( " Completed msiGetCDMSVariable microservice. " );
	freeAllDataObjInfo (dsetPathObjInfo);
	cdmsLog( " Memory freed, returning.... " );
    return (rei->status);
}

int msiTransferCDMSVariable( msParam_t *mspDatasetPath, msParam_t *mspVarName, msParam_t *mspRoi, msParam_t *mspOutputResc, msParam_t *mspPhysOutputPath, msParam_t *mspObjectPath, ruleExecInfo_t *rei)
{
	RE_TEST_MACRO( "    Calling msiGetCDMSVariable");
	cdmsLog( "Calling msiGetCDMSVariable" );
	char *strVarName = parseMspForStr(mspVarName);
	char *strRoi = parseMspForStr(mspRoi);
	rsComm_t *rsComm = rei->rsComm;
	int err_code = 0, status;

	char* strZone = rsComm->myEnv.rodsZone;
	char* userName = rsComm->myEnv.rodsUserName;
	char* strRodsHome = rsComm->myEnv.rodsHome;
	char* strRodsEnvFile = getRodsEnvFileName();
    dataObjInp_t dataObjInp, *myDataObjInp;
    if ((rei->status = parseMspForDataObjInp ( mspDatasetPath, &dataObjInp, &myDataObjInp, 0)) < 0) {
        rodsLogAndErrorMsg (LOG_ERROR, &rsComm->rError, rei->status, "msiGetCDMSVariable: input mspDatasetPath error. status = %d", rei->status);
        return (rei->status);
    }
	dataObjInfo_t *dsetPathObjInfo = NULL;
	status = getDataObjInfo (rsComm, myDataObjInp, &dsetPathObjInfo, NULL, 1);
    if (status < 0) {
        rodsLog ( LOG_ERROR, "msiGetCDMSVariable: getDataObjInfo for %s", myDataObjInp->objPath );
        return (status);
    }
	char *strDatasetPhysicalPath = dsetPathObjInfo->filePath;

	// Call the python microservice with the parameters
	cdmsLog( " Call function transferVariable( '%s', '%s', '%s', '%s' ) ", userName, strDatasetPhysicalPath, strVarName, strRoi );
	char *path = transferVariable( userName, strDatasetPhysicalPath, strVarName, strRoi );
	if ( path == NULL ) {
        rodsLog ( LOG_ERROR, "Error Getting Variable." );
        return (status);
	}
	char *file_path = strdup( path );
	char* outputResc = strdup( rsComm->myEnv.rodsDefResource );

//	/* Register the subsetted file in iRods */
//	dataObjInp_t newDataObjInp;					/* for collection registration */
//	memset(&newDataObjInp, 0, sizeof(dataObjInp_t));
///
//	addKeyVal (&newDataObjInp.condInput, COLLECTION_KW, "");
//	addKeyVal (&newDataObjInp.condInput, DEST_RESC_NAME_KW, outputResc );
//	addKeyVal (&newDataObjInp.condInput, FILE_PATH_KW, path );
//
	char new_file_name[ MAX_NAME_LEN ], obj_path[ MAX_NAME_LEN ];
	for ( int is=0; is <= strlen( path); is++ ) {
		new_file_name[is] = ( path[is] == '/' ) ?  '_' : path[is];
	}

	/* Similarly, reconstruct iRODS path of (target) new collection */
	snprintf( obj_path, MAX_NAME_LEN, "%s/uvcdat-transfer/%s", strRodsHome, new_file_name );


	/* Registration happens here */

//	char new_file_name[ MAX_NAME_LEN ];
//	for ( int is=0; is <= strlen( path); is++ ) {
//		new_file_name[is] = ( path[is] == '/' ) ?  '_' : path[is];
//	}
//
//	char data_object_path[ MAX_NAME_LEN ];
//	snprintf( data_object_path, MAX_NAME_LEN, "%s/uvcdat-transfer/", strRodsHome );
//
//    rodsPathInp_t rodsPathInp;
//    rodsArguments_t myRodsArgs;
//	rodsPath_t rp1, rp2, rp3;
//	memset((void*)&myRodsArgs, 0, sizeof(rodsArguments_t));
//	memset((void*)&rodsPathInp, 0, sizeof(rodsPathInp_t));
//	memset((void*)&rp1, 0, sizeof(rodsPath_t));
//	memset((void*)&rp2, 0, sizeof(rodsPath_t));
//	rodsPathInp.srcPath = &rp1;
//	rodsPathInp.destPath = &rp2;
//	rodsPathInp.numSrc = 1;
//	rodsPathInp.srcPath->objType = LOCAL_FILE_T;
//	rodsPathInp.srcPath->objState = EXIST_ST;
//	sprintf(rodsPathInp.srcPath->outPath, "%s", path);
//	rodsPathInp.destPath->objType = COLL_OBJ_T;
//	rodsPathInp.destPath->objState = EXIST_ST;
//	sprintf(rodsPathInp.destPath->outPath, "%s/uvcdat-transfer/%s", strRodsHome, new_file_name );
//
//    status = putUtil ( &rsComm, &rsComm->myEnv, &myRodsArgs, &rodsPathInp);
//
//    printErrorStack( rsComm->rError );

//	rei->status = rsPhyPathReg (rei->rsComm, &newDataObjInp);
//	if (rei->status < 0) {
//		rodsLog (LOG_ERROR, "msiTransferCDMSVariable: rsPhyPathReg('%s/uvcdat-transfer') failed with status %d", strRodsHome, rei->status);
//		return rei->status;
//	}

	dataObjInp_t oldDataObjInp;
	memset ( &oldDataObjInp, 0, sizeof (dataObjInp));
	addKeyVal  (&oldDataObjInp.condInput, FORCE_FLAG_KW, "" );
    snprintf ( oldDataObjInp.objPath, MAX_NAME_LEN, "%s", obj_path );
    rsDataObjUnlink ( rsComm, &oldDataObjInp );

	rei->status = err_code;
	if (rei->status >= 0) {
		cdmsLog( " Filling MsParam, resc = %s, path = %s ", outputResc, file_path );
		mspOutputResc->inOutStruct = (void*) outputResc;
		mspOutputResc->type = strdup(STR_MS_T);
		mspPhysOutputPath->inOutStruct = (void*) file_path;
		mspPhysOutputPath->type = strdup(STR_MS_T);
		mspObjectPath->inOutStruct = (void*) strdup(obj_path);
		mspObjectPath->type = strdup(STR_MS_T);
	} else {
		rodsLogAndErrorMsg( LOG_ERROR, &rsComm->rError, rei->status, "msiGetCDMSVariable failed, status = %d", rei->status );
	}

	cdmsLog( " Completed msiGetCDMSVariable microservice. File path = %s, Object path = %s ", file_path, obj_path );
	freeAllDataObjInfo (dsetPathObjInfo);
	cdmsLog( " Memory freed, returning.... " );
    return (rei->status);
}


int msiPythonInitialize(ruleExecInfo_t *rei) {
	cdmsLog( "msiPythonInitialize" );
	pythonInitialize();
	return 0;
}

int msiPythonFinalize(ruleExecInfo_t *rei) {
	cdmsLog( "msiPythonFinalize" );
	pythonFinalize();
	return 0;
}

int cdms_rsNccfGetVara (int ncid,  nccfGetVarInp_t * nccfGetVarInp, nccfGetVarOut_t ** nccfGetVarOut)
{
    int status;
    nc_type xtypep;
    int nlat = 0;
    int nlon = 0;
    void *data = NULL;
    char *dataType_PI;
    int typeSize;
    int dataLen;

    if (nccfGetVarInp == NULL || nccfGetVarOut == NULL)
        return USER__NULL_INPUT_ERR;

    *nccfGetVarOut = NULL;
    status = nc_inq_vartype (ncid, nccfGetVarInp->varid, &xtypep);
    if (status != NC_NOERR) return NETCDF_INQ_VARS_ERR - status;

    switch (xtypep) {
      case NC_CHAR:
      case NC_BYTE:
      case NC_UBYTE:
	typeSize = sizeof (char);
	dataType_PI = "charDataArray_PI";
	break;
      case NC_STRING:
	typeSize = sizeof (char *);
	dataType_PI = "strDataArray_PI";
	break;
      case NC_INT:
      case NC_UINT:
      case NC_FLOAT:
	dataType_PI = "intDataArray_PI";
	typeSize = sizeof (int);
	break;
      case NC_SHORT:
      case NC_USHORT:
        dataType_PI = "int16DataArray_PI";
        typeSize = sizeof (short);
        break;
      case NC_INT64:
      case NC_UINT64:
      case NC_DOUBLE:
	typeSize = sizeof (double);
	dataType_PI = "int64DataArray_PI";
	break;
      default:
        rodsLog (LOG_ERROR,
          "cdms_rsNccfGetVara: Unknow dataType %d", xtypep);
        return (NETCDF_INVALID_DATA_TYPE);
    }
    if (nccfGetVarInp->maxOutArrayLen <= 0) {
	dataLen = DEF_OUT_ARRAY_BUF_SIZE;
    } else {
	dataLen = typeSize * nccfGetVarInp->maxOutArrayLen;
	if (dataLen < MIN_OUT_ARRAY_BUF_SIZE) {
	    dataLen = MIN_OUT_ARRAY_BUF_SIZE;
	} else if (dataLen > MAX_OUT_ARRAY_BUF_SIZE) {
            rodsLog (LOG_ERROR,
              "cdms_rsNccfGetVara: dataLen %d larger than MAX_OUT_ARRAY_BUF_SIZE",
	      dataLen);
	    return NETCDF_VARS_DATA_TOO_BIG;
	}
    }
    *nccfGetVarOut = (nccfGetVarOut_t *) calloc (1, sizeof (nccfGetVarOut_t));
    (*nccfGetVarOut)->dataArray = (dataArray_t *) calloc
      (1, sizeof (dataArray_t));

    (*nccfGetVarOut)->dataArray->buf = data = calloc (1, dataLen);

    status = nccf_get_vara (ncid, nccfGetVarInp->varid,
      nccfGetVarInp->latRange, &nlat, nccfGetVarInp->lonRange, &nlon,
      nccfGetVarInp->lvlIndex,  nccfGetVarInp->timestep, data);

    if (status == NC_NOERR) {
	(*nccfGetVarOut)->dataArray->len = nlat * nlon;
	/* sanity check. It's too late */
	if ((*nccfGetVarOut)->dataArray->len * typeSize > dataLen) {
            rodsLog (LOG_ERROR,
              "cdms_rsNccfGetVara:  nccf_get_vara outlen %d > alloc len %d.",
	      (*nccfGetVarOut)->dataArray->len, dataLen);
            freeNccfGetVarOut (nccfGetVarOut);
	    return NETCDF_VARS_DATA_TOO_BIG;
	}
	(*nccfGetVarOut)->nlat = nlat;
	(*nccfGetVarOut)->nlon = nlon;
	rstrcpy ((*nccfGetVarOut)->dataType_PI, dataType_PI, NAME_LEN);
        (*nccfGetVarOut)->dataArray->type = xtypep;
    } else {
        freeNccfGetVarOut (nccfGetVarOut);
        rodsLog (LOG_ERROR,
          "cdms_rsNccfGetVara:  nccf_get_vara err varid %d dataType %d. %s ",
          nccfGetVarInp->varid, xtypep, nc_strerror(status));
        status = NETCDF_GET_VARS_ERR - status;
    }
    return status;
}

int cdms_rsNccfGetVara (rsComm_t *rsComm,  nccfGetVarInp_t * nccfGetVarInp, nccfGetVarOut_t ** nccfGetVarOut)
{
    int remoteFlag;
    rodsServerHost_t *rodsServerHost = NULL;
    int l1descInx;
    nccfGetVarInp_t myNccfGetVarInp;
    int status = 0;

    if (getValByKey (&nccfGetVarInp->condInput, NATIVE_NETCDF_CALL_KW) !=
      NULL) {
        status = cdms_rsNccfGetVara (nccfGetVarInp->ncid, nccfGetVarInp,
          nccfGetVarOut);
        return status;
    }
    l1descInx = nccfGetVarInp->ncid;
    if (l1descInx < 2 || l1descInx >= NUM_L1_DESC) {
        rodsLog (LOG_ERROR,
          "cdms_rsNccfGetVara: l1descInx %d out of range",
          l1descInx);
        return (SYS_FILE_DESC_OUT_OF_RANGE);
    }
    if (L1desc[l1descInx].inuseFlag != FD_INUSE) return BAD_INPUT_DESC_INDEX;
    if (L1desc[l1descInx].remoteZoneHost != NULL) {
        myNccfGetVarInp = *nccfGetVarInp;
        myNccfGetVarInp.ncid = L1desc[l1descInx].remoteL1descInx;

        /* cross zone operation */
        status = rcNccfGetVara (L1desc[l1descInx].remoteZoneHost->conn,
          &myNccfGetVarInp, nccfGetVarOut);
    } else {
        remoteFlag = resoAndConnHostByDataObjInfo (rsComm,
          L1desc[l1descInx].dataObjInfo, &rodsServerHost);
        if (remoteFlag < 0) {
            return (remoteFlag);
        } else if (remoteFlag == LOCAL_HOST) {
            status = cdms_rsNccfGetVara (L1desc[l1descInx].l3descInx,
              nccfGetVarInp, nccfGetVarOut);
            if (status < 0) {
                return status;
            }
        } else {
            /* execute it remotely */
            myNccfGetVarInp = *nccfGetVarInp;
            myNccfGetVarInp.ncid = L1desc[l1descInx].l3descInx;
            addKeyVal (&myNccfGetVarInp.condInput, NATIVE_NETCDF_CALL_KW, "");
            status = rcNccfGetVara (rodsServerHost->conn, &myNccfGetVarInp,
              nccfGetVarOut);
            clearKeyVal (&myNccfGetVarInp.condInput);
            if (status < 0) {
                rodsLog (LOG_ERROR,
                  "cdms_rsNccfGetVara: rcNccfGetVara %d for %s error, status = %d",
                  L1desc[l1descInx].l3descInx,
                  L1desc[l1descInx].dataObjInfo->objPath, status);
                return (status);
            }
        }
    }
    return status;
}

/**
 * \fn msiNccfGetVara (msParam_t *ncidParam, msParam_t *varidParam, msParam_t *lvlIndexParam, msParam_t *timestepParam,  msParam_t *latRange0Param, msParam_t *latRange1Param, msParam_t *lonRange0Param, msParam_t *lonRange1Param, msParam_t *maxOutArrayLenParam, msParam_t *outParam, ruleExecInfo_t *rei)
 *
**/
int msiCdmsGetVara (msParam_t *ncidParam, msParam_t *varidParam, msParam_t *lvlIndexParam, msParam_t *timestepParam, msParam_t *latRange0Param, msParam_t *latRange1Param, msParam_t *lonRange0Param, msParam_t *lonRange1Param, msParam_t *maxOutArrayLenParam, msParam_t *outParam, ruleExecInfo_t *rei) {
    rsComm_t *rsComm;
    nccfGetVarInp_t nccfGetVarInp;
    nccfGetVarOut_t *nccfGetVarOut = NULL;

    if (rei == NULL || rei->rsComm == NULL) {
      rodsLog (LOG_ERROR, "msiNccfGetVara: input rei or rsComm is NULL");
      return (SYS_INTERNAL_NULL_INPUT_ERR);
    }
    rsComm = rei->rsComm;

    if (ncidParam == NULL) {
        rodsLog (LOG_ERROR, "msiNccfGetVara: input ncidParam is NULL");
        return (SYS_INTERNAL_NULL_INPUT_ERR);
    }

    /* parse for dataType or nccfGetVarInp_t */
    rei->status = parseMspForNccfGetVarInp (ncidParam, &nccfGetVarInp);

    if (rei->status < 0) return rei->status;

    if (varidParam != NULL) {
        /* parse for varid */
        nccfGetVarInp.varid = parseMspForPosInt (varidParam);
        if (nccfGetVarInp.varid < 0) return nccfGetVarInp.varid;
    }

    if (lvlIndexParam != NULL) {
        /* parse for ndim */
        nccfGetVarInp.lvlIndex = parseMspForPosInt (lvlIndexParam);
        if (nccfGetVarInp.lvlIndex < 0) return nccfGetVarInp.lvlIndex;
    }

    if (timestepParam != NULL) {
        /* parse for ndim */
        nccfGetVarInp.timestep = parseMspForPosInt (timestepParam);
        if (nccfGetVarInp.timestep < 0) return nccfGetVarInp.timestep;
    }

    if (latRange0Param != NULL) {
        rei->status = parseMspForFloat (latRange0Param,  &nccfGetVarInp.latRange[0]);
        if (rei->status < 0) return rei->status;
    }

    if (latRange1Param != NULL) {
    	rei->status = parseMspForFloat (latRange1Param,  &nccfGetVarInp.latRange[1]);
    	if (rei->status < 0) return rei->status;
    }

    if (lonRange0Param != NULL) {
    	rei->status  = parseMspForFloat (lonRange0Param,  &nccfGetVarInp.lonRange[0]);
    	if (rei->status < 0) return rei->status;
    }

    if (lonRange1Param != NULL) {
    	rei->status = parseMspForFloat (lonRange1Param,  &nccfGetVarInp.lonRange[1]);
    	if (rei->status < 0) return rei->status;
    }

    if (maxOutArrayLenParam != NULL) {
        /* parse for maxOutArrayLen */
        nccfGetVarInp.maxOutArrayLen = parseMspForPosInt (maxOutArrayLenParam);
        if (nccfGetVarInp.maxOutArrayLen < 0) { return nccfGetVarInp.maxOutArrayLen; }
    }

    rei->status = cdms_rsNccfGetVara (rsComm, &nccfGetVarInp, &nccfGetVarOut);
    clearKeyVal (&nccfGetVarInp.condInput);
    if (rei->status >= 0) {
    	fillMsParam (outParam, NULL, NccfGetVarOut_MS_T, nccfGetVarOut, NULL);
    } else {
        rodsLogAndErrorMsg (LOG_ERROR, &rsComm->rError, rei->status, "msiNccfGetVara: cdms_rsNccfGetVara failed, status = %d", rei->status);
    }

    return (rei->status);
}

