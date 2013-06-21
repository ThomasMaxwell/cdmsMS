#ifndef CDMSMS_H
#define CDMSMS_H

#include "apiHeaderAll.h"
#include "objMetaOpr.h"
#include "miscUtil.h"

/*
int msiRodsPython(msParam_t *script_path, msParam_t *func_name, msParam_t *rule_name, ruleExecInfo_t *rei);
int msiRodsPython1(msParam_t *script_path, msParam_t *func_name, msParam_t *rule_name, msParam_t *param1, ruleExecInfo_t *rei);
int msiRodsPython2(msParam_t *script_path, msParam_t *func_name, msParam_t *rule_name, msParam_t *param1, msParam_t *param2, ruleExecInfo_t *rei);
int msiLocalPython(msParam_t *irods_script_path, msParam_t *func_name, msParam_t *rule_name, ruleExecInfo_t *rei);
int msiLocalPython1(msParam_t *irods_script_path, msParam_t *func_name, msParam_t *rule_name, msParam_t *param1, ruleExecInfo_t *rei);
int msiLocalPython2(msParam_t *irods_script_path, msParam_t *func_name, msParam_t *rule_name, msParam_t *param1, msParam_t *param2, ruleExecInfo_t *rei);
int msiExecPython(msParam_t *python_code, ruleExecInfo_t *rei);
int msiImportPythonZip(msParam_t *zip_path, ruleExecInfo_t *rei);
*/

int msiGetCDMSVariable( msParam_t *dataset_path, msParam_t *var_name, msParam_t *roi, msParam_t *result, ruleExecInfo_t *rei );
int msiTransferCDMSVariable( msParam_t *dataset_path, msParam_t *var_name, msParam_t *roi, msParam_t *mspOutputResc, msParam_t *mspPhysOutputPath, msParam_t *mspObjectPath, ruleExecInfo_t *rei );
int msiPythonInitialize( ruleExecInfo_t *rei );
int msiPythonFinalize( ruleExecInfo_t *rei );
int msiCDMSTest( msParam_t *test_string, ruleExecInfo_t *rei);
int msiCdmsGetVara( msParam_t *ncidParam, msParam_t *varidParam, msParam_t *lvlIndexParam, msParam_t *timestepParam, msParam_t *latRange0Param, msParam_t *latRange1Param, msParam_t *lonRange0Param, msParam_t *lonRange1Param, msParam_t *maxOutArrayLenParam, msParam_t *outParam, ruleExecInfo_t *rei );

#endif	/* CDMSMS */
