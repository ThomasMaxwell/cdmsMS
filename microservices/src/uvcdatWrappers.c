/*
 ============================================================================
 Name        : uvcdatWrappers.c
 Author      : Thomas Maxwell
 Description : Wrap uvcdat functionality in C for embedding in irods microservices.
 ============================================================================
 */
#include <Python.h>
#include "arrayobject.h"

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

void* getVariable( char* user_name, char* dataset_path, char* var_name, char* roi )
{
    PyObject *pScript, *pModule, *pFunc, *pArgs;
	char buffer[250];

    char* script_path = "/Developer/Projects/iRODS/src-3.2/modules/cdms/python/CDMS_DataServices.py";
    char* method_name = "getCDMSVariable";

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
    return NULL;
}

char* transferVariable( char* user_name, char* dataset_path, char* var_name, char* roi )
{
    PyObject *pScript, *pModule, *pFunc, *pArgs;
	char buffer[250];

    char* script_path = "/Developer/Projects/iRODS/src-3.2/modules/cdms/python/CDMS_DataServices.py";
    char* method_name = "transferCDMSVariable";

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
            pArgs = PyTuple_New( 4 );
            PyTuple_SetItem( pArgs, 0, PyString_FromString( user_name ) );
            PyTuple_SetItem( pArgs, 1, PyString_FromString( dataset_path ) );
            PyTuple_SetItem( pArgs, 2, PyString_FromString( var_name ) );
            PyTuple_SetItem( pArgs, 3, PyString_FromString( roi ) );
            PyObject *path = (PyObject*) PyObject_CallObject( pFunc, pArgs );
            Py_DECREF(pArgs);
            if (path != NULL) {
                return PyBytes_AsString(path);
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

int getNDim( void* arr ) {
	return (int) PyArray_NDIM( arr );
}

void* getRawData( void* arr ) {
	return PyArray_DATA( arr );
}

int getDims( void* arr, int* dims ) {
	npy_intp* np_dims = PyArray_DIMS( arr );
	int i;
	for( i=0; i<getNDim( arr ); i++ ) {
		dims[i] = (int)np_dims[i];
	}
	return 0;
}

int getSize1( void* arr ) {
	return (int) PyArray_Size( (PyObject *) arr );
}

int getSize2( void* arr ) {
	return (int) PyArray_Size( (PyObject *) arr );
//	return (int) (((PyArrayObject *)(arr))->nd));
}

int getSize( void* arr ) {
	npy_intp* np_dims = PyArray_DIMS( arr );
	int i, size = 1, nd = getNDim( arr );
	for( i=0; i<nd; i++ ) {
		size = size * (int)np_dims[i];
	}
	return size;
}

int getStrides( void* arr, int* strides ) {
	npy_intp* np_dims = PyArray_STRIDES( arr );
	int i;
	for( i=0; i<getNDim( arr ); i++ ) {
		strides[i] = (int)np_dims[i];
	}
	return 0;
}

int getItemSize( void* arr ) {
	return (int) PyArray_ITEMSIZE( arr );
}

int getNBytes( void* arr ) {
	return (int) PyArray_NBYTES( arr );
}

int getType( void* arr ) {
	return (int) PyArray_TYPE( arr );
}

int isFloat( void* arr ) {
	return (int) PyArray_ISFLOAT( arr );
}

int isInteger( void* arr ) {
	return (int) PyArray_ISINTEGER( arr );
}

int isString( void* arr ) {
	return (int) PyArray_ISSTRING( arr );
}

int isSigned( void* arr ) {
	return (int) PyArray_ISSIGNED( arr );
}

const char* getTypeDesc( void* arr ) {
	PyArray_Descr *desc = PyArray_DESCR( arr );
	if( desc != NULL ) {
		switch( desc->kind ) {
			case 'b' : return " boolean";
			case 'i' : return " signed integer";
			case 'u' : return " unsigned integer";
			case 'f' : return " floating point";
			case 'c' : return " complex floating point";
			case 'S' : return " 8-bit character string";
			case 'U' : return " 32-bit/character unicode string.";
			case 'V' : return " arbitrary.";
		}
	}
	return "unknown.";
}






