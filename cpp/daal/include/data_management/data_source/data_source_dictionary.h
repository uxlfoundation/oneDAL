/* file: data_source_dictionary.h */
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
//  Implementation of a data source dictionary.
//--
*/

#ifndef __DATA_SOURCE_DICTIONARY_H__
#define __DATA_SOURCE_DICTIONARY_H__

#include <map>
#include <string>

#include "services/internal/collection.h"
#include "data_management/features/defines.h"
#include "data_management/data/data_dictionary.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @ingroup data_sources
 * @{
 */

class CategoricalFeatureDictionary : public std::map<std::string, std::pair<int, int> >
{};
typedef services::SharedPtr<CategoricalFeatureDictionary> CategoricalFeatureDictionaryPtr;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATASOURCEFEATURE"></a>
 *  \brief Data structure that describes the Data Source feature
 */
class DataSourceFeature : public SerializationIface
{
public:
    NumericTableFeature ntFeature;
    size_t name_length;
    char * name;

    CategoricalFeatureDictionary * cat_dict;

public:
    /**
     *  Constructor of a data feature
     */
    DataSourceFeature() : name_length(0), name(NULL), cat_dict(NULL) {}

    /**
     *  Copy constructor for a data feature
     */
    DataSourceFeature(const DataSourceFeature & other) { assign(other); }

    /**
     *  Assigment operator for a data feature
     */
    DataSourceFeature & operator=(const DataSourceFeature & other) { return assign(other); }

    /** \private */
    virtual ~DataSourceFeature()
    {
        if (_catDictPtr.get() != cat_dict)
        {
            delete cat_dict;
        }
    }

    /**
     * Returns the name of the feature
     */
    services::String getFeatureName() const { return services::String(name); }

    /**
     *  Gets a categorical features dictionary
     *  \return Pointer to the categorical features dictionary
     */
    CategoricalFeatureDictionary * getCategoricalDictionary()
    {
        if (!cat_dict)
        {
            cat_dict    = new CategoricalFeatureDictionary();
            _catDictPtr = CategoricalFeatureDictionaryPtr(cat_dict);
        }

        return cat_dict;
    }

    void setCategoricalDictionary(const CategoricalFeatureDictionaryPtr & dictionary)
    {
        if (_catDictPtr.get() != cat_dict)
        {
            delete cat_dict;
            cat_dict = NULL;
        }

        _catDictPtr = dictionary;
        cat_dict    = dictionary.get();
    }

    /**
     *  Specifies the name of a data feature
     *  \param[in]  featureName  Name of the data feature
     */
    void setFeatureName(const services::String & featureName)
    {
        _name = featureName;
        synchRawAndStringNames();
    }

    /**
     *  Fills the class based on a specified type
     *  \tparam  T  Name of the data feature
     */
    template <typename T>
    void setType()
    {
        ntFeature.setType<T>();
    }

    /** \private */
    services::Status serializeImpl(InputDataArchive * arch) override { return serialImpl<InputDataArchive, false>(arch); }

    /** \private */
    services::Status deserializeImpl(const OutputDataArchive * arch) override { return serialImpl<const OutputDataArchive, true>(arch); }

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        services::Status status;

        arch->setObj(&ntFeature);
        arch->set(name_length);

        if (onDeserialize)
        {
            if (name_length > 0)
            {
                _name = services::String(name_length);
                synchRawAndStringNames();
            }
        }

        arch->set(name, name_length);

        const int categoricalFeatureDictionaryFlag = (cat_dict != 0);
        arch->set(categoricalFeatureDictionaryFlag);

        if (categoricalFeatureDictionaryFlag)
        {
            if (onDeserialize)
            {
                /* Make sure that dictionary is allocated */
                getCategoricalDictionary();
                /* Make sure that dictionary is empty */
                cat_dict->clear();
            }

            size_t size = cat_dict->size();
            arch->set(size);

            if (onDeserialize)
            {
                const size_t initialBuffSize = 10;
                services::internal::PrimitiveCollection<char> buff(initialBuffSize, &status);
                DAAL_CHECK_STATUS_VAR(status);

                for (size_t i = 0; i < size; i++)
                {
                    size_t catNameLen = 0;
                    int catV1         = 0;
                    int catV2         = 0;

                    arch->set(catNameLen);
                    if (catNameLen > buff.size())
                    {
                        DAAL_CHECK_STATUS(status, buff.reallocate(catNameLen));
                    }
                    arch->set(buff.data(), catNameLen);
                    arch->set(catV1);
                    arch->set(catV2);

                    (*cat_dict)[std::string(buff.data(), catNameLen)] = std::pair<int, int>(catV1, catV2);
                }
            }
            else
            {
                typedef CategoricalFeatureDictionary::iterator it_type;

                for (it_type it = cat_dict->begin(); it != cat_dict->end(); it++)
                {
                    const std::string & catName = it->first;
                    size_t catNameLen           = catName.size();
                    int catV1                   = it->second.first;
                    int catV2                   = it->second.second;

                    arch->set(catNameLen);
                    arch->set(catName.c_str(), catNameLen);
                    arch->set(catV1);
                    arch->set(catV2);
                }
            }
        }
        else
        {
            cat_dict    = NULL;
            _catDictPtr = CategoricalFeatureDictionaryPtr();
        }

        return status;
    }

    DAAL_EXPORT virtual int getSerializationTag() const override { return SERIALIZATION_DATAFEATURE_NT_ID; }

    features::IndexNumType getIndexType() const { return ntFeature.indexType; }

private:
    DataSourceFeature & assign(const DataSourceFeature & other)
    {
        _name       = other._name;
        _catDictPtr = other._catDictPtr;
        ntFeature   = other.ntFeature;
        cat_dict    = other.cat_dict;

        if (other.name == other._name.c_str())
        {
            synchRawAndStringNames();
        }
        else
        {
            name        = other.name;
            name_length = other.name_length;
        }

        return *this;
    }

    void synchRawAndStringNames()
    {
        name_length = _name.length();
        name        = const_cast<char *>(_name.c_str());
    }

private:
    services::String _name;
    CategoricalFeatureDictionaryPtr _catDictPtr;
};

typedef Dictionary<DataSourceFeature, SERIALIZATION_DATADICTIONARY_DS_ID> DataSourceDictionary;
typedef services::SharedPtr<DataSourceDictionary> DataSourceDictionaryPtr;
/** @} */

} // namespace interface1

using interface1::CategoricalFeatureDictionary;
using interface1::CategoricalFeatureDictionaryPtr;
using interface1::DataSourceFeature;
using interface1::DataSourceDictionary;
using interface1::DataSourceDictionaryPtr;

} // namespace data_management
} // namespace daal

#endif
