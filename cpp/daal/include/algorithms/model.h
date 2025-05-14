/* file: model.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Data model classes declarations
//--
*/

#ifndef __MODEL_H__
#define __MODEL_H__

#include "data_management/data/data_archive.h"
#include "services/base.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @addtogroup base_algorithms
 * @{
 */
struct ValidationMetricIface
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MODEL"></a>
 * \brief The base class for the classes that represent the models, such as linear_regression::Model
 *        or svm::Model
 * \note  The current version of the library does not support generation of validation metrics
 *        for the models
 */
class Model : public data_management::SerializationIface
{
public:
    /** Default constructor */
    Model() {}

    virtual ~Model() DAAL_C11_OVERRIDE {}

    /**
     * \copydoc daal::data_management::interface1::SerializationIface::getSerializationTag()
     */
    DAAL_EXPORT int getSerializationTag() const DAAL_C11_OVERRIDE { return 0; }

protected:
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * /*arch*/)
    {
        return services::Status();
    }

    DECLARE_SERIALIZABLE_IMPL()
};
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface1
using interface1::ValidationMetricIface;
using interface1::Model;
using interface1::ModelPtr;

} // namespace algorithms
} // namespace daal
#endif
