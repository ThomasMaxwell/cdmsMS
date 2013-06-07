//#include <time.h>
//#include <cstdlib>
//#include <iostream>
//#include <unistd.h>

#include <cstdarg>

#include "rods.h"
#include "cdmsMS.h"
#include "dataObjOpr.h"
#include "uvcdatWrappers.h"

#define BIG_STR 2000

void cdmsLog( char *formatStr, ... ) {
	char bigString[BIG_STR];
	va_list ap;
	va_start(ap, formatStr);
	vsnprintf(bigString, BIG_STR-1, formatStr, ap);
	va_end(ap);
	rodsLog(LOG_NOTICE, "<-------------------------> CDMS Message <------------------------->\n     %s  ", bigString );
}


//int readTextFile(rsComm_t *rsComm, char* inPath, bytesBuf_t *data)
//{
//	/* ********** *
//	 * Initialize *
//	 * ********** */
//
//	dataObjInp_t openParam;
//	openedDataObjInp_t closeParam;
//	openedDataObjInp_t readParam;
//	openedDataObjInp_t seekParam;
//	fileLseekOut_t* seekResult = NULL;
//	int fd = -1;
//	int fileWasOpened = FALSE;
//	int fileLength = 0;
//	int status;
//
//	memset(&openParam,  0, sizeof(dataObjInp_t));
//	memset(&seekParam,  0, sizeof(openedDataObjInp_t));
//	memset(&readParam,  0, sizeof(openedDataObjInp_t));
//	memset(&closeParam, 0, sizeof(openedDataObjInp_t));
//	memset(data,        0, sizeof(bytesBuf_t));
//
//
//	/* ************* *
//	 * Open the file *
//	 * ************* */
//
//	// Set the parameters for the open call
//	strcpy(openParam.objPath, inPath);
//	openParam.openFlags = O_RDONLY;
//
//	status = rsDataObjOpen(rsComm, &openParam);
//
//	if (status < 0) { return status; }
//	fd = status;
//	fileWasOpened = TRUE;
//
//
//	/* ************************ *
//     * Looking for the filesize *
//	 * ************************ */
//
//	// Go to the end of the file
//	seekParam.l1descInx = fd;
//	seekParam.offset  = 0;
//	seekParam.whence  = SEEK_END;
//
//	status = rsDataObjLseek(rsComm, &seekParam, &seekResult);
//
//	if (status < 0) {
//		// Try to close the file we opened, ignoring errors
//		if (fileWasOpened) {
//			closeParam.l1descInx = fd;
//			rsDataObjClose(rsComm, &closeParam);
//		}
//		return status;
//	}
//	fileLength = seekResult->offset;
//
//	// Reset to the start for the read
//	seekParam.offset  = 0;
//	seekParam.whence  = SEEK_SET;
//
//	status = rsDataObjLseek(rsComm, &seekParam, &seekResult);
//
//	if (status < 0) {
//		// Try to close the file we opened, ignoring errors
//		if (fileWasOpened) {
//			closeParam.l1descInx = fd;
//			rsDataObjClose(rsComm, &closeParam);
//		}
//		return status;
//	}
//
//
//	/* ************* *
//	 * Read the file *
//	 * ************* */
//
//
//	// Set the parameters for the open call
//	readParam.l1descInx = fd;
//	readParam.len       = fileLength;
//	data->len           = fileLength;
//	data->buf           = (void*)malloc(fileLength);
//
//	// Read the file
//	status = rsDataObjRead(rsComm, &readParam, data);
//	if (status < 0)	{
//		free((char*)data->buf);
//		// Try to close the file we opened, ignoring errors
//		if (fileWasOpened) {
//			closeParam.l1descInx = fd;
//			rsDataObjClose(rsComm, &closeParam);
//		}
//		return status;
//	}
//
//	/* ************** *
//	 * Close the file *
//	 * ************** */
//
//	// Close the file we opened
//	if (fileWasOpened) {
//		closeParam.l1descInx = fd;
//
//		status = rsDataObjClose(rsComm, &closeParam);
//		if (status < 0) {
//			free((char*)data->buf);
//			return status;
//		}
//	}
//
//	return 0;
//}


int msiGetCDMSVariable( msParam_t *mspDatasetPath, msParam_t *mspVarName, msParam_t *mspRoi, msParam_t *mspResult, ruleExecInfo_t *rei)
{
	RE_TEST_MACRO( "    Calling msiGetCDMSVariable");

//	PyGILState_STATE gstate = PyGILState_Ensure();
//	char *strFunctionName = "getCDMSVariable";
//	char *strScriptName = "CDMS_DataServices.py";

	char *strVarName = parseMspForStr(mspVarName);
	char *strRoi = parseMspForStr(mspRoi);
	rsComm_t *rsComm = rei->rsComm;
	bytesBuf_t inData;
	int err_code = 0, status;

	rodsEnv env;
    status = getRodsEnv (&env);
    if (status < 0) {
        rodsLogError (LOG_ERROR, status, "main: getRodsEnv error. ");
        return status;
    }
	char* strZone = env.rodsZone;
	char* strRodsHome = env.rodsHome;
	char* strRodsEnvFile = getRodsEnvFileName();
//	char strScriptPath[BIG_STR];
//	snprintf( strScriptPath, BIG_STR, "/%s/home/public/Microservices/%s", strZone, strScriptName );
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

//	// Read the script from iRODS, get a string
//	status = readTextFile(rsComm, strScriptPath, &inData);
//	if (status < 0) {
//		rodsLogAndErrorMsg(LOG_ERROR, &rsComm->rError, status,  "%s:  could not read file, status = %d", strScriptPath, status);
//		return USER_FILE_DOES_NOT_EXIST;
//	}
//
//	// Execute the script (It will load the functions defined in the script in the global dictionary)
//	char tmpStr[inData.len+1];
//	snprintf(tmpStr, inData.len, "%s", (char *)inData.buf);
//	cdmsLog( " Execute script: \n %s \n\n *rodsEnvFile = %s \n", tmpStr, strRodsEnvFile );
//	err_code = PyRun_SimpleString(tmpStr);
//	if (err_code == -1) {
//		PyErr_Print();
//		err_code = INVALID_OBJECT_TYPE;
//		rei->status = err_code;
//		return err_code;
//	}
//	// Get a reference to the main module and global dictionary
//	PyObject *pModule = PyImport_AddModule("__main__");
//	PyObject *pDict = PyModule_GetDict(pModule);
//	// Get a reference to the function we want to call
//	PyObject *pFunc = PyDict_GetItemString(pDict, strFunctionName);
//	if (!pFunc) {
//		PyErr_Print();
//		err_code = NO_MICROSERVICE_FOUND_ERR;
//		rei->status = err_code;
//		return err_code;
//	}
//	if (!PyCallable_Check(pFunc) ) {
//		err_code = NO_MICROSERVICE_FOUND_ERR;
//		rei->status = err_code;
//		rodsLogAndErrorMsg(LOG_ERROR, &rsComm->rError, err_code,  "Function is not callable: %s ", strFunctionName );
//		return err_code;
//	}
	// Call the python microservice with the parameters
	cdmsLog( " Call function getVariable( '%s', '%s', '%s' ) ", strDatasetPhysicalPath, strVarName, strRoi );
	void *arr = getVariable( strDatasetPhysicalPath, strVarName, strRoi );
	cdmsLog( " Completed function call. " );


//	PyObject* pystrDatasetPhysicalPath = PyString_FromString( strDatasetPhysicalPath );
//	PyObject* pystrVarName = PyString_FromString( strVarName );
//	PyObject* pystrRoi = PyString_FromString( strRoi );
//	PyObject *arr = (PyObject*) PyObject_CallFunctionObjArgs( pFunc, pystrDatasetPhysicalPath, pystrVarName, pystrRoi );
//	PyArrayObject *arr = (PyArrayObject*) PyObject_CallFunctionObjArgs( pFunc, pystrDatasetPhysicalPath, pystrVarName, pystrRoi );
//	if ( arr == NULL ) {
//		// if CallFunction fails rv is NULL. This is an error in the
//		// python script (wrong name, syntax error, ...
//		// The PyErr_Print will print it in rodsLog
//		PyErr_Print();
//		err_code = INVALID_OBJECT_TYPE; // not the best one but it exists
//		return err_code;
//	}
//
//	void *raw_data = PyArray_DATA( arr );
//	npy_intp *dims = PyArray_DIMS( arr );
//	npy_intp len = PyArray_SIZE( arr );
//	int ndim = PyArray_NDIM( arr );
//	PyArray_Descr *desc = PyArray_DESCR( arr );
//	npy_intp *strides = PyArray_STRIDES( arr );
//
//	ncGetVarOut_t *ncGetVarOut = (ncGetVarOut_t *) calloc (1, sizeof (ncGetVarOut_t));
//    ncGetVarOut->dataArray = (dataArray_t *) calloc (1, sizeof (dataArray_t));
//    ncGetVarOut->dataArray->len = len;
//    int dtype = setDataArrayType( ncGetVarOut, desc->kind );
//    if( dtype == NETCDF_INVALID_DATA_TYPE ) { err_code = INVALID_OBJECT_TYPE; }
//    ncGetVarOut->dataArray->buf = raw_data;
//
//    rei->status = err_code;
//    if (rei->status >= 0) {
//    	fillMsParam ( mspResult, NULL, NcGetVarOut_MS_T, &ncGetVarOut, NULL );
//    } else {
//    	rodsLogAndErrorMsg( LOG_ERROR, &rsComm->rError, rei->status, "msiGetCDMSVariable failed, status = %d", rei->status );
//    }
	freeAllDataObjInfo (dsetPathObjInfo);
    return (rei->status);
}

int msiPythonInitialize(ruleExecInfo_t *rei) {
//	pythonInitialize();
	return 0;
}

int msiPythonFinalize(ruleExecInfo_t *rei) {
//	pythonFinalize()
	return 0;
}


