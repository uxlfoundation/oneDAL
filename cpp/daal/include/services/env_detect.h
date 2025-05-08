/* file: env_detect.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
* Copyright contributors to the oneDAL project
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
//  Implementation of the class used for environment detection.
//--
*/

#ifndef __ENV_DETECT_H__
#define __ENV_DETECT_H__

#include "services/base.h"
#include "services/daal_defines.h"
#include "services/error_handling.h"
#include "services/cpu_type.h"

namespace daal
{
/**
 * @defgroup env_detect Managing the Computational Environment
 * \brief Provides methods to interact with the environment, including processor detection and control by the number of threads.
 * @ingroup services
 * @{
 */
namespace services
{
namespace interface1
{
/**
 * <a name="DAAL-CLASS-SERVICES__ENVIRONMENT"></a>
 * \brief Class that provides methods to interact with the environment, including processor detection and control by the number of threads
 */
class DAAL_EXPORT Environment : public Base
{
public:
    /**
     * <a name="DAAL-CLASS-_ENVSTRUCT"></a>
     * \brief The environment structure
     */
    typedef struct _envStruct
    {
        bool cpuid_init_flag;
        size_t cpuid;
    } env;

    /**
     *  Returns the environment instance
     *  \return The environment instance
     */
    static Environment * getInstance();

    /**
     *  Decreases the instance counter
     *  \return The return code
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED static int freeInstance();

    /**
     *  <a name="DAAL-ENUM-SERVICES__CPUTYPEENABLE"></a>
     *  \brief CPU types
     *  \DAAL_DEPRECATED
     */
    enum CpuTypeEnable
    {
        cpu_default = 0, /*!< Default processor type */

#if defined(TARGET_X86_64)
        avx512 = 2 /*!< Intel(R) Xeon(R) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) \DAAL_DEPRECATED */
#elif defined(TARGET_ARM)
        sve = 2, /*!< ARM(R) processors based on Arm's Scalable Vector Extension (SVE) */
#elif defined(TARGET_RISCV64)
        rv64 = 1
#endif
    };

    /**
     *  Detects the processor type
     *  \param[in] enable  An enabling flag \DAAL_DEPRECATED
     *  \return The CPU ID
     */
    int getCpuId(int enable = cpu_default);

    /**
     *  Restrict dispatching to the required code path
     *  \param[in] cpuid  CPU ID
     *  \return  CPU ID if success; ErrorCpuInvalid if cpuid value is out of CpuType enum
     */
    int setCpuId(int cpuid);

    /**
     *  Enable dispatching for new Intel(R) architectures
     *  \param[in] enable  An enabling flag
     *  \return  CPU ID
     *  \DAAL_DEPRECATED
     */
    int enableInstructionsSet(int enable);

    /**
     * <a name="DAAL-ENUM-SERVICES__LIBRARYTHREADINGTYPE"></a>
     * The threading mode of the library
     */
    enum LibraryThreadingType
    {
        MultiThreaded = 0 /*!< Multi-threaded mode */
    };

    /**
     *  Sets the threading mode on Windows*
     *  \param[in] type  The threading mode of the library
     */
    void setDynamicLibraryThreadingTypeOnWindows(LibraryThreadingType type);

    /**
     *  Sets the number of threads to use
     *  \param[in] numThreads   The number of threads
     */
    void setNumberOfThreads(const size_t numThreads);

    /**
     *  Enables thread pinning
     *  \param[in] enableThreadPinningFlag   Flag to thread pinning enable
     */
    void enableThreadPinning(bool enableThreadPinningFlag = true);

    /**
     *  Returns the number of used threads
     *  \return The number of used threads
     */
    size_t getNumberOfThreads() const;

    /**
     * Limits the amount of memory of the given type available to internal function calls
     * \param[in] type   Memory type
     * \param[in] limit  Limit in megabytes
     */
    int setMemoryLimit(MemType type, size_t limit);

private:
    Environment();
    Environment(const Environment & e);
    Environment & operator=(const Environment &);
    ~Environment();

    void _cpu_detect(int);
    void initNumberOfThreads();

    env _env;
    // Pointer to the oneapi::tbb::task_scheduler_handle class object, global for oneDAL.
    // The oneapi::tbb::task_scheduler_handle and the oneapi::tbb::finalize function
    // allow user to wait for completion of worker threads.
    void * _schedulerHandle;
    void * _globalControl;
};
} // namespace interface1

using interface1::Environment;

} // namespace services
/** @} */
} // namespace daal
#endif
