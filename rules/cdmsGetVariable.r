cdmsGetVariable {
	msiPythonInitialize();
	msiGetCDMSVariable( *dsetPath, *varName, *roi, *result );
	msiPythonFinalize();
}
INPUT *dsetPath="/uvcdatZone/home/rods/comp-ECMWF/ecmwf.xml", *varName="Temperature", *roi="(lat=(-90,90))"
OUTPUT *result
