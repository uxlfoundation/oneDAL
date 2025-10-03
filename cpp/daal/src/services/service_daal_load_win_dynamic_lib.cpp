/* file: service_daal_load_win_dynamic_lib.cpp */
/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of safe load library functionality for Windows.
//--
*/

#if defined(_WIN32) || defined(_WIN64)

    #include "services/daal_defines.h"
    #include <windows.h>

    #if !defined(DAAL_CHECK_DLL_SIG)
        #define DAAL_LOAD_DLL(name) LoadLibrary(name)
    #else
        #define DAAL_LOAD_DLL(name) _DAALLoadLibrary(name)

        #include <stdio.h>
        #include <Softpub.h>
        #include <wincrypt.h>
        #include <wintrust.h>

        #pragma comment(lib, "Wintrust.lib")

static HMODULE WINAPI _DAALLoadLibrary(LPCTSTR filename)
{
    TCHAR PathBuf[MAX_PATH];
    DWORD rv;
    BOOL rv1;
    HMODULE rv2 = NULL;

    // References:
    // "Dynamic-Link Library Security" - https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-security
    // "Dynamic-Link Library Search Order" - https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order?redirectedfrom=MSDN
    // "SearchPath function" - https://docs.microsoft.com/en-us/windows/win32/api/processenv/nf-processenv-searchpathw
    // "SetSearchPathMode function" - https://docs.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setsearchpathmode
    // "SetDllDirectory function" - https://docs.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setdlldirectorya
    // "LoadLibraryExA function" - https://docs.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa

    // Exclude current directory from the search path
    rv1 = SetDllDirectoryA("");
    if (0 == rv1)
    {
        printf("Intel oneDAL FATAL ERROR: Cannot exclude current directory from search path.\n");
        return NULL;
    }

    rv2 = LoadLibraryExA(filename, NULL, DONT_RESOLVE_DLL_REFERENCES);
    if (NULL == rv2)
    {
        printf("Intel oneDAL FATAL ERROR: Cannot find/load library %s.\n", filename);
        return NULL;
    }

    rv = GetModuleFileNameA(rv2, PathBuf, MAX_PATH);
    if (0 == rv)
    {
        printf("Intel oneDAL FATAL ERROR: Cannot find module %s in memory.\n", filename);
        return NULL;
    }

    FreeLibrary(rv2);
    rv2 = NULL;

    size_t strLength      = strnlen(PathBuf, MAX_PATH) + 1;
    wchar_t * wPathBuf    = new wchar_t[strLength];
    size_t convertedChars = 0;
    mbstowcs_s(&convertedChars, wPathBuf, strLength, PathBuf, _TRUNCATE);

    LONG sverif;
    DWORD lerr;
    WINTRUST_FILE_INFO fdata;
    GUID pgActionID;
    WINTRUST_DATA pWVTData;

    fdata.cbStruct       = sizeof(WINTRUST_FILE_INFO);
    fdata.pcwszFilePath  = wPathBuf;
    fdata.hFile          = NULL;
    fdata.pgKnownSubject = NULL;

    pgActionID = WINTRUST_ACTION_GENERIC_VERIFY_V2;

    pWVTData.cbStruct            = sizeof(WINTRUST_DATA);
    pWVTData.pPolicyCallbackData = NULL;
    pWVTData.pSIPClientData      = NULL;
    pWVTData.dwUIChoice          = WTD_UI_NONE;
    pWVTData.fdwRevocationChecks = WTD_REVOKE_NONE;
    pWVTData.dwUnionChoice       = WTD_CHOICE_FILE;
    pWVTData.pFile               = &fdata;
    pWVTData.dwStateAction       = WTD_STATEACTION_VERIFY;
    pWVTData.hWVTStateData       = NULL;
    pWVTData.pwszURLReference    = NULL;
    pWVTData.dwProvFlags         = 0;
    pWVTData.dwUIContext         = WTD_UICONTEXT_EXECUTE;
    pWVTData.pSignatureSettings  = NULL;

    sverif = WinVerifyTrust((HWND)INVALID_HANDLE_VALUE, &pgActionID, &pWVTData);

    switch (sverif)
    {
    case TRUST_E_NOSIGNATURE:
        lerr = GetLastError();
        if (TRUST_E_NOSIGNATURE == lerr || TRUST_E_SUBJECT_FORM_UNKNOWN == lerr || TRUST_E_PROVIDER_UNKNOWN == lerr)
        {
            printf("Intel oneDAL FATAL ERROR: %s is not signed.\n", filename);
        }
        else
        {
            printf("Intel oneDAL FATAL ERROR: An unknown error occurred trying to verify the signature of the %s.\n", filename);
        }
        break;

    case TRUST_E_EXPLICIT_DISTRUST: printf("Intel oneDAL FATAL ERROR: The signature/publisher of %s is disallowed.\n", filename); break;

    case CERT_E_UNTRUSTEDROOT:
        printf(
            "Intel oneDAL FATAL ERROR: The certificate chain of %s processed, but terminated in a root certificate which is not trusted by the trust "
            "provider.\n",
            filename);
        break;

    case ERROR_SUCCESS: break;

    case TRUST_E_SUBJECT_NOT_TRUSTED: printf("Intel oneDAL FATAL ERROR: The signature of %s in not trusted.\n", filename); break;

    case CRYPT_E_SECURITY_SETTINGS:
        printf("Intel oneDAL FATAL ERROR: %s. The subject hash or publisher was not explicitly trusted and user trust was not allowed "
               "(CRYPT_E_SECURITY_SETTINGS).\n",
               filename);
        break;

    default: printf("Intel oneDAL FATAL ERROR: %s. Error code is 0x%x.\n", filename, (unsigned int)sverif); break;
    }

    pWVTData.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(NULL, &pgActionID, &pWVTData);
    delete[] wPathBuf;

    if (ERROR_SUCCESS != sverif)
    {
        return NULL;
    }

    rv2 = LoadLibraryA(PathBuf);

    // Restore current directory from the search path
    SetDllDirectory(NULL);

    return rv2;
}
    #endif

DAAL_EXPORT HMODULE _daal_load_win_dynamic_lib(LPCTSTR filename)
{
    return DAAL_LOAD_DLL(filename);
}

#endif // defined(_WIN32) || defined(_WIN64)
