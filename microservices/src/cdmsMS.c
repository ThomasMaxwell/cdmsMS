#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <unistd.h>

#include "rods.h"
#include "cdmsMS.h"
#include <Python.h>
#include "arrayobject.h"
#include "ncGetVarsByType.h"

//#include "PyMsParam.h"

#define BIG_STR 2000
rodsEnv env;

//char* getFilename(char* path) {
//	char str[strlen(path)];
//	char *pch;
//	char *filename = (char*)malloc(sizeof(char)*strlen(path));
//
//	strcpy(str, path);
//
//	pch = strtok (str,"/");
//	while (pch != NULL)
//	{
//		strcpy(filename, pch);
//		pch = strtok (NULL, "/");
//	}
//
//	return strtok (filename, ".");
//}
//
//char* getBasename(char* path) {
//	char str[strlen(path)];
//	char * pch;
//	char *basename = (char*)malloc(sizeof(char)*strlen(path));
//
//	strcpy(str, path);
//	strcpy(basename, "/");
//
//	pch = strtok (str,"/");
//	while (pch != NULL)
//	{
//		char elem[50];
//		strcpy(elem, pch);
//		pch = strtok (NULL, "/");
//		// For the last element
//		if (pch != NULL) {
//			strcat(basename, elem);
//			strcat(basename, "/");
//		}
//	}
//	return basename;
//}


int readTextFile(rsComm_t *rsComm, char* inPath, bytesBuf_t *data)
{
	/* ********** *
	 * Initialize *
	 * ********** */

	dataObjInp_t openParam;
	openedDataObjInp_t closeParam;
	openedDataObjInp_t readParam;
	openedDataObjInp_t seekParam;
	fileLseekOut_t* seekResult = NULL;
	int fd = -1;
	int fileWasOpened = FALSE;
	int fileLength = 0;
	int status;

	memset(&openParam,  0, sizeof(dataObjInp_t));
	memset(&seekParam,  0, sizeof(openedDataObjInp_t));
	memset(&readParam,  0, sizeof(openedDataObjInp_t));
	memset(&closeParam, 0, sizeof(openedDataObjInp_t));
	memset(data,        0, sizeof(bytesBuf_t));


	/* ************* *
	 * Open the file *
	 * ************* */

	// Set the parameters for the open call
	strcpy(openParam.objPath, inPath);
	openParam.openFlags = O_RDONLY;

	status = rsDataObjOpen(rsComm, &openParam);

	if (status < 0) { return status; }
	fd = status;
	fileWasOpened = TRUE;


	/* ************************ *
     * Looking for the filesize *
	 * ************************ */

	// Go to the end of the file
	seekParam.l1descInx = fd;
	seekParam.offset  = 0;
	seekParam.whence  = SEEK_END;

	status = rsDataObjLseek(rsComm, &seekParam, &seekResult);

	if (status < 0) {
		// Try to close the file we opened, ignoring errors
		if (fileWasOpened) {
			closeParam.l1descInx = fd;
			rsDataObjClose(rsComm, &closeParam);
		}
		return status;
	}
	fileLength = seekResult->offset;

	// Reset to the start for the read
	seekParam.offset  = 0;
	seekParam.whence  = SEEK_SET;

	status = rsDataObjLseek(rsComm, &seekParam, &seekResult);

	if (status < 0) {
		// Try to close the file we opened, ignoring errors
		if (fileWasOpened) {
			closeParam.l1descInx = fd;
			rsDataObjClose(rsComm, &closeParam);
		}
		return status;
	}


	/* ************* *
	 * Read the file *
	 * ************* */


	// Set the parameters for the open call
	readParam.l1descInx = fd;
	readParam.len       = fileLength;
	data->len           = fileLength;
	data->buf           = (void*)malloc(fileLength);

	// Read the file
	status = rsDataObjRead(rsComm, &readParam, data);
	if (status < 0)	{
		free((char*)data->buf);
		// Try to close the file we opened, ignoring errors
		if (fileWasOpened) {
			closeParam.l1descInx = fd;
			rsDataObjClose(rsComm, &closeParam);
		}
		return status;
	}

	/* ************** *
	 * Close the file *
	 * ************** */

	// Close the file we opened
	if (fileWasOpened) {
		closeParam.l1descInx = fd;

		status = rsDataObjClose(rsComm, &closeParam);
		if (status < 0) {
			free((char*)data->buf);
			return status;
		}
	}

	return 0;
}

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

int msiGetCDMSVariable( msParam_t *dataset_path, msParam_t *var_name, msParam_t *roi, msParam_t *result, ruleExecInfo_t *rei)
{
	RE_TEST_MACRO( "    Calling msiGetCDMSVariable");

	char *str_dataset_path = parseMspForStr(dataset_path);
	char *str_var_name = parseMspForStr(var_name);
	char *str_roi = parseMspForStr(roi);
	rsComm_t *rsComm = rei->rsComm;
	bytesBuf_t inData;
	int err_code = 0, status;
	char str_script_path[BIG_STR];

	char* zone = env.rodsZone;
	snprintf( str_script_path, BIG_STR, "/%s/home/public/Microservices/CDMS_DataServices.py", zone );
	char *str_func_name = "getCDMSVariable";

	// Read the script from iRODS, get a string
	status = readTextFile(rsComm, str_script_path, &inData);
	if (status < 0) {
		rodsLogAndErrorMsg(LOG_ERROR, &rsComm->rError, status,  "%s:  could not read file, status = %d", str_script_path, status);
		return USER_FILE_DOES_NOT_EXIST;
	}

	// Execute the script (It will load the functions defined in the script in the global dictionary)
	char tmpStr[inData.len];
	snprintf(tmpStr, inData.len, "%s", (char *)inData.buf);
	err_code = PyRun_SimpleString(tmpStr);
	if (err_code == -1) {
		PyErr_Print();
		err_code = INVALID_OBJECT_TYPE;
		rei->status = err_code;
		return err_code;
	}
	// Get a reference to the main module and global dictionary
	PyObject *pModule = PyImport_AddModule("__main__");
	PyObject *pDict = PyModule_GetDict(pModule);
	// Get a reference to the function we want to call
	PyObject *pFunc = PyDict_GetItemString(pDict, str_func_name);
	if (!pFunc) {
		PyErr_Print();
		err_code = NO_MICROSERVICE_FOUND_ERR;
		rei->status = err_code;
		return err_code;
	}
	// Call the python microservice with the parameters
	PyArrayObject *arr = (PyArrayObject*) PyObject_CallFunction(pFunc, "sss", str_dataset_path, str_var_name, str_roi );

	if ( arr == NULL ) {
		// if CallFunction fails rv is NULL. This is an error in the
		// python script (wrong name, syntax error, ...
		// The PyErr_Print will print it in rodsLog
		PyErr_Print();
		err_code = INVALID_OBJECT_TYPE; // not the best one but it exists
	}

	void *raw_data = PyArray_DATA( arr );
	npy_intp *dims = PyArray_DIMS( arr );
	npy_intp len = PyArray_SIZE( arr );
	int ndim = PyArray_NDIM( arr );
	PyArray_Descr *desc = PyArray_DESCR( arr );
	npy_intp *strides = PyArray_STRIDES( arr );

	ncGetVarOut_t *ncGetVarOut = (ncGetVarOut_t *) calloc (1, sizeof (ncGetVarOut_t));
    ncGetVarOut->dataArray = (dataArray_t *) calloc (1, sizeof (dataArray_t));
    ncGetVarOut->dataArray->len = len;
    int dtype = setDataArrayType( ncGetVarOut, desc->kind );
    if( dtype == NETCDF_INVALID_DATA_TYPE ) { err_code = INVALID_OBJECT_TYPE; }
    ncGetVarOut->dataArray->buf = raw_data;

    rei->status = err_code;
    if (rei->status >= 0) {
    	fillMsParam ( result, NULL, NcGetVarOut_MS_T, &ncGetVarOut, NULL );
    } else {
    	rodsLogAndErrorMsg( LOG_ERROR, &rsComm->rError, rei->status, "msiGetCDMSVariable failed, status = %d", rei->status );
    }
    return (rei->status);
}

int msiPythonInitialize(ruleExecInfo_t *rei) {
	// Initialize the python interpreter
	if (!Py_IsInitialized()) {
	    int status = getRodsEnv (&env);
	    if (status < 0) {
	        rodsLogError (LOG_ERROR, status, "main: getRodsEnv error. ");
	        return -1;
	    }
	    Py_Initialize();
//		PyRun_SimpleString("call_history = {}");
//		PyRun_SimpleString("imported_zip_packages = []");
	}
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



