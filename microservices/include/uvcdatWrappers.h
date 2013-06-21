/*
 ============================================================================
 Name        : uvcdatWrappers.h
 Author      : Thomas Maxwell
 Description : Wrap uvcdat functionality in C for embedding in irods microservices.
 ============================================================================
 */

void* getVariable( char* userName, char* dataset_path, char* var_name, char* roi );

char* transferVariable( char* userName, char* dataset_path, char* var_name, char* roi );

void* getRawData( void* arr );

int getDims( void* arr, int* dims );

int getSize( void* arr );

int getNDim( void* arr );

int getStrides( void* arr, int* strides ) ;

int getType( void* arr ) ;

int isFloat( void* arr ) ;

int isInteger( void* arr ) ;

int isString( void* arr ) ;

int isSigned( void* arr ) ;

int getItemSize( void* arr ) ;

int getNBytes( void* arr ) ;

const char* getTypeDesc( void* arr );

void pythonInitialize();

void pythonFinalize();




