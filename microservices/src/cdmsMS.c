#include "Python.h"
//#include <time.h>
//#include <cstdlib>
//#include <iostream>
//#include <unistd.h>

#include "rods.h"
#include "cdmsMS.h"
#include "arrayobject.h"
#include "ncGetVarsByType.h"
#include "dataObjOpr.h"

#define BIG_STR 2000

void cdmsLog( char *formatStr, ... ) {
	char bigString[BIG_STR];
	va_list ap;
	va_start(ap, formatStr);
	vsnprintf(bigString, BIG_STR-1, formatStr, ap);
	va_end(ap);
	rodsLog(LOG_NOTICE, "<-------------------------> CDMS Message <------------------------->\n     %s  ", bigString );
}

char* getFilename(char* path) {
	char str[strlen(path)];
	char *pch;
	char *filename = (char*)malloc(sizeof(char)*strlen(path));

	strcpy(str, path);

	pch = strtok (str,"/");
	while (pch != NULL)
	{
		strcpy(filename, pch);
		pch = strtok (NULL, "/");
	}

	return strtok (filename, ".");
}

char* getBasename(char* path) {
	char str[strlen(path)];
	char * pch;
	char *basename = (char*)malloc(sizeof(char)*strlen(path));

	strcpy(str, path);
	strcpy(basename, "/");

	pch = strtok (str,"/");
	while (pch != NULL)
	{
		char elem[50];
		strcpy(elem, pch);
		pch = strtok (NULL, "/");
		// For the last element
		if (pch != NULL) {
			strcat(basename, elem);
			strcat(basename, "/");
		}
	}
	return basename;
}

PyArrayObject* getVariable( char* dataset_path, char* var_name, char* roi )
{
    PyObject *pScript, *pModule, *pFunc, *pArgs;
	char buffer[250];
    int i, err_code;

    char* script_path = "/Developer/Projects/iRODS/src-3.2/modules/cdms/python/CDMS_DataServices.py";
    char* method_name = "getCDMSVariable";

    Py_Initialize();
	PyRun_SimpleString("import sys");
	sprintf( buffer, "sys.path.insert(0, '%s')", getBasename(script_path) );
	PyRun_SimpleString(buffer);

	// Get the pointer of the function you want to call
	pScript = PyString_FromString( getFilename(script_path) );
	pModule = PyImport_Import(pScript);
	Py_DECREF(pScript);
    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, method_name );
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New( 3 );
            PyTuple_SetItem( pArgs, 0, PyString_FromString( dataset_path ) );
            PyTuple_SetItem( pArgs, 1, PyString_FromString( var_name ) );
            PyTuple_SetItem( pArgs, 2, PyString_FromString( roi ) );
            PyArrayObject *arr = (PyArrayObject*) PyObject_CallObject( pFunc, pArgs );
            Py_DECREF(pArgs);
            if (arr != NULL) {
                return arr;
            }
            else {
            	if( PyErr_Occurred() ) { PyErr_Print(); }
                fprintf(stderr,"Call failed\n");
            }
        }
        else {
            if( PyErr_Occurred() ) { PyErr_Print(); }
            fprintf(stderr, "Cannot find function \"%s\"\n", method_name );
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", script_path );
    }
    Py_Finalize();
    return NULL;
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

//	char desc.kind:  A ÔbÕ represents Boolean, a ÔiÕ represents signed integer, a ÔuÕ represents unsigned integer, ÔfÕ represents floating point,
//  ÔcÕ represents complex floating point, ÔSÕ represents 8-bit character string, ÔUÕ represents 32-bit/character unicode string, and ÔVÕ repesents arbitrary.

int setDataArrayType( ncGetVarOut_t *ncGetVarOut, char kind ) {
	int type = NETCDF_INVALID_DATA_TYPE;
	switch ( kind ) {
	   case 'S':
		   type = NC_CHAR;
		   rstrcpy (ncGetVarOut->dataType_PI, "charDataArray_PI", NAME_LEN);
		   break;
	   case 'i':
		   type = NC_INT;
		   rstrcpy (ncGetVarOut->dataType_PI, "intDataArray_PI", NAME_LEN);
		   break;
	   case 'u':
		   type = NC_UINT;
		   rstrcpy (ncGetVarOut->dataType_PI, "intDataArray_PI", NAME_LEN);
		   break;
	   case 'f':
		   type = NC_FLOAT;
		   rstrcpy (ncGetVarOut->dataType_PI, "intDataArray_PI", NAME_LEN);
		   break;
	   default:
		 rodsLog (LOG_ERROR, "msiGetCDMSVariable:setDataArrayType- Unknow dataType: '%c'", kind );
	 }
	 ncGetVarOut->dataArray->type = type;
	return type;
}

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
	PyArrayObject *arr = getVariable( strDatasetPhysicalPath, strVarName, strRoi );
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
	// Initialize the python interpreter
	if (!Py_IsInitialized()) {
	    Py_Initialize();
//		PyRun_SimpleString("call_history = {}");
//		PyRun_SimpleString("imported_zip_packages = []");
	}
	return 0;
}

int msiCDMSTest(msParam_t *test_string, ruleExecInfo_t *rei) {
	char *str_test_string = parseMspForStr(test_string);
	rodsLog( LOG_NOTICE, "<-------------------------> Testing msiCDMS[1]: test string = '%s' <-------------------------> ", str_test_string) ;
	return 0;
}


int msiPythonFinalize(ruleExecInfo_t *rei) {
	// Undo all initializations made by Py_Initialize()
	// There's a bug in python 2 when you initialize/finalize the interpreter
	// several times in the same process. This happens for the hooks which
	// are dealt by the same irodsAgent process.
	// Without finalize, I hope that when the thread is disposed of, the memory
	// is cleaned.

	Py_Finalize();
//	PyRun_SimpleString("import os");
//	PyRun_SimpleString("for f in imported_zip_packages:os.remove(f)");
	return 0;
}


