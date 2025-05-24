/* file: data_dictionary.h */
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
//  Implementation of a data dictionary.
//--
*/

#ifndef __DATA_DICTIONARY_H__
#define __DATA_DICTIONARY_H__

#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_archive.h"

#include "data_management/features/defines.h"
#include "data_management/features/internal/helpers.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @defgroup data_dictionary Data Dictionaries
 * \brief Contains classes that represent a dictionary of a data set and provide methods to work with the data dictionary
 * @ingroup data_management
 * @{
 */

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__NUMERICTABLEFEATURE"></a>
 *  \brief Data structure describes the Numeric Table feature
 */
class NumericTableFeature : public SerializationIface
{
public:
    features::IndexNumType indexType;
    features::PMMLNumType pmmlType;
    features::FeatureType featureType;
    size_t typeSize;
    size_t categoryNumber;

public:
    /**
     *  Constructor of a data feature
     */
    NumericTableFeature()
    {
        indexType      = features::DAAL_OTHER_T;
        pmmlType       = features::DAAL_GEN_UNKNOWN;
        featureType    = features::DAAL_CONTINUOUS;
        typeSize       = 0;
        categoryNumber = 0;
    }

    /**
     * Copy constructor for a data feature
     */
    NumericTableFeature(const NumericTableFeature & f)
    {
        indexType      = f.indexType;
        pmmlType       = f.pmmlType;
        featureType    = f.featureType;
        typeSize       = f.typeSize;
        categoryNumber = f.categoryNumber;
    }

    /**
     *  Copy operator for a data feature
     */
    NumericTableFeature & operator=(const NumericTableFeature & f)
    {
        if (this == &f) return *this;

        indexType      = f.indexType;
        pmmlType       = f.pmmlType;
        featureType    = f.featureType;
        typeSize       = f.typeSize;
        categoryNumber = f.categoryNumber;

        return *this;
    }

    virtual ~NumericTableFeature() {}

    /**
     *  Fills the class based on a specified type
     *  \tparam  T  Name of the data feature
     */
    template <typename T>
    void setType()
    {
        typeSize  = sizeof(T);
        indexType = features::internal::getIndexNumType<T>();
        pmmlType  = features::internal::getPMMLNumType<T>();
    }

    /** \private */
    services::Status serializeImpl(InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    /** \private */
    services::Status deserializeImpl(const OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->set(pmmlType);
        arch->set(featureType);
        arch->set(typeSize);
        arch->set(categoryNumber);
        arch->set(indexType);

        return services::Status();
    }

    DAAL_EXPORT virtual int getSerializationTag() const DAAL_C11_OVERRIDE { return SERIALIZATION_DATAFEATURE_NT_ID; }

    features::IndexNumType getIndexType() const { return indexType; }
};

/** \private */
class DictionaryIface
{
public:
    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__ALLOCATIONFLAG"></a>
     * \brief Enumeration to specify whether the Data Dictionary contains equal features
     */
    enum FeaturesEqual
    {
        notEqual, /*!< Data Dictionary contains different features */
        equal     /*!< Data Dictionary contains equal features */
    };
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DICTIONARY"></a>
 *  \brief Class that represents a dictionary of a data set
 *  and provides methods to work with the data dictionary
 */
template <typename Feature, int SerializationTag>
class DAAL_EXPORT Dictionary : public SerializationIface, public DictionaryIface
{
public:
    static int serializationTag() { return SerializationTag; }
    virtual int getSerializationTag() const DAAL_C11_OVERRIDE { return serializationTag(); }

    /**
     *  Constructor of a data dictionary
     *  \param[in]  nfeat  Number of features in the table
     *  \param[in]  featuresEqual Flag specifying that all features have equal types and properties
     *  \DAAL_DEPRECATED_USE{ Dictionary::create }
     */
    Dictionary(size_t nfeat, FeaturesEqual featuresEqual = notEqual)
        : _nfeat(0), _featuresEqual(featuresEqual), _dict(0), _errors(new services::KernelErrorCollection())
    {
        if (nfeat)
        {
            setNumberOfFeatures(nfeat);
        }
    }

    /**
     *  Constructs a data dictionary
     *  \param[in]  nfeat  Number of features in the table
     *  \param[in]  featuresEqual Flag specifying that all features have equal types and properties
     *  \param[in]  stat Status of the dictionary construction
     *  \return data dictionary
     */
    static services::SharedPtr<Dictionary> create(size_t nfeat, FeaturesEqual featuresEqual = notEqual, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_IMPL_EX(Dictionary, nfeat, featuresEqual);
    }

    /**
     *  Default constructor of a data dictionary
     *  \DAAL_DEPRECATED_USE{ Dictionary::create }
     */
    Dictionary() : _nfeat(0), _dict(0), _featuresEqual(DictionaryIface::notEqual), _errors(new services::KernelErrorCollection()) {}

    /**
     *  Constructs a default data dictionary
     *  \param[in]  stat Status of the dictionary construction
     *  \return data dictionary
     */
    static services::SharedPtr<Dictionary> create(services::Status * stat = NULL) { DAAL_DEFAULT_CREATE_IMPL(Dictionary); }

    /** \private */
    virtual ~Dictionary() { resetDictionary(); }

    /**
     *  Resets a dictionary and sets the number of features to 0
     */
    services::Status resetDictionary()
    {
        if (_dict)
        {
            delete[] _dict;
            _dict = NULL;
        }
        _nfeat = 0;
        return services::Status();
    }

    /**
     *  Sets all features of a dictionary to the same type
     *  \param[in]  defaultFeature  Default feature class to which to set all features
     */
    virtual services::Status setAllFeatures(const Feature & defaultFeature)
    {
        if (_featuresEqual == DictionaryIface::equal)
        {
            if (_nfeat > 0)
            {
                _dict[0] = defaultFeature;
            }
        }
        else
        {
            for (size_t i = 0; i < _nfeat; i++)
            {
                _dict[i] = defaultFeature;
            }
        }
        return services::Status();
    }

    /**
     *  Sets all features of a dictionary to the same type
     *  \tparam  featureType  Default feature type to which to set all features
     */
    template <typename featureType>
    services::Status setAllFeatures()
    {
        Feature defaultFeature;
        defaultFeature.template setType<featureType>();
        return setAllFeatures(defaultFeature);
    }

    /**
     *  Sets the number of features
     *  \param[in]  numberOfFeatures  Number of features
     */
    virtual services::Status setNumberOfFeatures(size_t numberOfFeatures)
    {
        resetDictionary();
        _nfeat = numberOfFeatures;
        if (_featuresEqual == DictionaryIface::equal)
        {
            _dict = new Feature[1];
            if (_dict == NULL)
            {
                return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
            }
        }
        else
        {
            _dict = new Feature[_nfeat];
            if (_dict == NULL)
            {
                return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
            }
        }
        return services::Status();
    }

    /**
     *  Returns the number of features
     *  \return Number of features
     */
    size_t getNumberOfFeatures() const { return _nfeat; }

    /**
     *  Returns the value of the featuresEqual flag
     *  \return Value of the featuresEqual flag
     */
    FeaturesEqual getFeaturesEqual() const { return _featuresEqual; }

    /**
     *  Returns a feature with a given index
     *  \param[in]  idx  Index of the feature
     *  \return Requested feature
     */
    Feature & operator[](const size_t idx)
    {
        if (_featuresEqual == DictionaryIface::equal)
        {
            return _dict[0];
        }
        else
        {
            return _dict[idx];
        }
    }

    /**
     *  \brief Adds a feature to a data dictionary
     *
     *  \param[in] feature  Data feature
     *  \param[in] idx      Index of the data feature
     *
     */
    services::Status setFeature(const Feature & feature, size_t idx)
    {
        if (idx >= _nfeat) return services::Status(services::ErrorIncorrectNumberOfFeatures);

        if (_featuresEqual == DictionaryIface::equal)
        {
            _dict[0] = feature;
        }
        else
        {
            _dict[idx] = feature;
        }
        return services::Status();
    }

    /**
     *  Adds a feature to a data dictionary
     *  \param[in] idx              Index of the data feature
     */
    template <typename T>
    services::Status setFeature(size_t idx)
    {
        Feature df;
        df.template setType<T>();
        return setFeature(df, idx);
    }

    services::Status checkDictionary() const
    {
        size_t nFeat = _nfeat;
        if (_featuresEqual == DictionaryIface::equal) nFeat = 1;

        for (size_t i = 0; i < nFeat; ++i)
        {
            if (_dict[i].getIndexType() == features::DAAL_OTHER_T)
            {
                return services::Status(services::ErrorUndefinedFeature);
            }
        }
        return services::Status();
    }

    /**
     * Returns errors during the computation
     * \return Errors during the computation
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::SharedPtr<services::KernelErrorCollection> getErrors() { return _errors; }

    /** \private */
    services::Status serializeImpl(InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    /** \private */
    services::Status deserializeImpl(const OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }

private:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->segmentHeader();

        arch->set(_nfeat);
        arch->set(_featuresEqual);

        if (onDeserialize)
        {
            size_t nfeat = _nfeat;
            _nfeat       = 0;
            setNumberOfFeatures(nfeat);
        }

        if (_featuresEqual)
        {
            arch->setObj(_dict, 1);
        }
        else
        {
            arch->setObj(_dict, _nfeat);
        }

        arch->segmentFooter();

        return services::Status();
    }

protected:
    size_t _nfeat;
    FeaturesEqual _featuresEqual;
    Feature * _dict;
    services::SharedPtr<services::KernelErrorCollection> _errors;

    Dictionary(size_t nfeat, FeaturesEqual featuresEqual, services::Status & st)
        : _nfeat(0), _featuresEqual(featuresEqual), _dict(0), _errors(new services::KernelErrorCollection())
    {
        if (nfeat)
        {
            st |= setNumberOfFeatures(nfeat);
        }
    }

    Dictionary(services::Status & /*st*/)
        : _nfeat(0), _featuresEqual(DictionaryIface::notEqual), _dict(0), _errors(new services::KernelErrorCollection())
    {}

private:
    Dictionary(const Dictionary &);
    Dictionary & operator=(const Dictionary &);
};
typedef Dictionary<NumericTableFeature, SERIALIZATION_DATADICTIONARY_NT_ID> NumericTableDictionary;
typedef services::SharedPtr<NumericTableDictionary> NumericTableDictionaryPtr;
typedef services::SharedPtr<NumericTableFeature> NumericTableFeaturePtr;
/** @} */

} // namespace interface1
using interface1::NumericTableFeature;
using interface1::NumericTableFeaturePtr;
using interface1::DictionaryIface;
using interface1::Dictionary;
using interface1::NumericTableDictionary;
using interface1::NumericTableDictionaryPtr;

} // namespace data_management
} // namespace daal
#endif
