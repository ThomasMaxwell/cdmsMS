/*
 ============================================================================
 Name        : uvcdatWrappers.c
 Author      : Thomas Maxwell
 Description : Wrap uvcdat functionality in C for embedding in irods microservices.
 ============================================================================
 */
#include <Python.h>
#include "arrayobject.h"
#include "ncGetVarsByType.h"

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

void* getVariable( char* dataset_path, char* var_name, char* roi )
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

void pythonInitialize() {
	// Initialize the python interpreter
	if (!Py_IsInitialized()) {
	    Py_Initialize();
//		PyRun_SimpleString("call_history = {}");
//		PyRun_SimpleString("imported_zip_packages = []");
	}
}

void pythonFinalize() {
	// Undo all initializations made by Py_Initialize()
	// There's a bug in python 2 when you initialize/finalize the interpreter
	// several times in the same process. This happens for the hooks which
	// are dealt by the same irodsAgent process.
	// Without finalize, I hope that when the thread is disposed of, the memory
	// is cleaned.

	Py_Finalize();
//	PyRun_SimpleString("import os");
//	PyRun_SimpleString("for f in imported_zip_packages:os.remove(f)");
}



