/*
 ============================================================================
 Name        : uvcdatWrappers.h
 Author      : Thomas Maxwell
 Description : Wrap uvcdat functionality in C for embedding in irods microservices.
 ============================================================================
 */

void* getVariable( char* dataset_path, char* var_name, char* roi );
void pythonFinalize();
void pythonInitialize();
