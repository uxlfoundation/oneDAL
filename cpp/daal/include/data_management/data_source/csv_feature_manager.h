/* file: csv_feature_manager.h */
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
//  Implementation of the CSV feature manager class.
//--
*/

#ifndef __CSV_FEATURE_MANAGER_H__
#define __CSV_FEATURE_MANAGER_H__

#include "data_management/data/numeric_table.h"
#include "data_management/features/shortcuts.h"
#include "data_management/data_source/data_source.h"
#include "data_management/data_source/internal/csv_feature_utils.h"
#include "data_management/data_source/modifiers/csv/shortcuts.h"
#include "data_management/data_source/modifiers/csv/internal/engine.h"

namespace daal
{
namespace data_management
{
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATUREAUXDATA"></a>
 *  \brief Structure for auxiliary data used for feature extraction.
 */
struct FeatureAuxData
{
    FeatureAuxData() : idx(0), wide(1), nCats(0), dsFeat(0), ntFeat(0), buffer() {}

    explicit FeatureAuxData(size_t index, DataSourceFeature * dataSourceFeature, NumericTableFeature * numericTableFeature)
        : idx(index), wide(1), nCats(0), dsFeat(dataSourceFeature), ntFeat(numericTableFeature), buffer()
    {}

    size_t idx;
    size_t wide;
    size_t nCats;
    DataSourceFeature * dsFeat;
    NumericTableFeature * ntFeat;
    std::string buffer;
};

typedef void (*functionT)(const char * word, FeatureAuxData & aux, DAAL_DATA_TYPE * arr);

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERIFACE"></a>
 *  \brief Abstract interface class that defines the interface for a features modifier
 */
class ModifierIface
{
public:
    virtual void apply(services::Collection<functionT> & funcList, services::Collection<FeatureAuxData> & auxVect) const = 0;

    virtual ~ModifierIface() {}

    static void contFunc(const char * word, FeatureAuxData & aux, DAAL_DATA_TYPE * arr);

    static void catFunc(const char * word, FeatureAuxData & aux, DAAL_DATA_TYPE * arr)
    {
        aux.buffer.assign(word);

        CategoricalFeatureDictionary * catDict    = aux.dsFeat->getCategoricalDictionary();
        CategoricalFeatureDictionary::iterator it = catDict->find(aux.buffer);

        if (it != catDict->end())
        {
            arr[aux.idx] = (DAAL_DATA_TYPE)it->second.first;
            it->second.second++;
        }
        else
        {
            int index = (int)(catDict->size());
            catDict->insert(std::pair<std::string, std::pair<int, int> >(aux.buffer, std::pair<int, int>(index, 1)));
            arr[aux.idx]               = (DAAL_DATA_TYPE)index;
            aux.ntFeat->categoryNumber = index + 1;
        }
    }

    static void nullFunc(const char * /*word*/, FeatureAuxData & /*aux*/, DAAL_DATA_TYPE * /*arr*/) {}

protected:
    template <class T>
    static void readNumeric(const char * text, T & f);

    static void binFunc(const char * word, FeatureAuxData & aux, DAAL_DATA_TYPE * arr)
    {
        aux.buffer.assign(word);

        CategoricalFeatureDictionary * catDict    = aux.dsFeat->getCategoricalDictionary();
        CategoricalFeatureDictionary::iterator it = catDict->find(aux.buffer);

        size_t index = 0;

        if (it != catDict->end())
        {
            index = it->second.first;
            it->second.second++;
        }
        else
        {
            index = catDict->size();
            catDict->insert(std::pair<std::string, std::pair<int, int> >(aux.buffer, std::pair<int, int>((int)index, 1)));
            aux.ntFeat->categoryNumber = index + 1;
        }

        size_t nCats = aux.nCats;

        for (size_t i = 0; i < nCats; i++)
        {
            arr[aux.idx + i] = (DAAL_DATA_TYPE)(i == index);
        }
    }
};

template <class T>
inline void ModifierIface::readNumeric(const char * text, T & f)
{
    f = daal::services::daal_string_to_float(text, 0);
}

template <>
inline void ModifierIface::readNumeric(const char * text, double & f)
{
    f = daal::services::daal_string_to_double(text, 0);
}

inline void ModifierIface::contFunc(const char * word, FeatureAuxData & aux, DAAL_DATA_TYPE * arr)
{
    DAAL_DATA_TYPE f;
    readNumeric<DAAL_DATA_TYPE>(word, f);
    arr[aux.idx] = f;
}

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__MAKECATEGORICAL"></a>
 *  \brief Methods of the class to set a feature categorical.
 */
class MakeCategorical : public ModifierIface
{
    size_t idx;

public:
    MakeCategorical(size_t idx) : idx(idx) {}

    virtual ~MakeCategorical() {}

    virtual void apply(services::Collection<functionT> & funcList, services::Collection<FeatureAuxData> & auxVect) const override
    {
        size_t nCols = funcList.size();

        if (idx < nCols)
        {
            funcList[idx] = catFunc;
            auxVect[idx].buffer.resize(1024);
        }
    }
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__MAKECATEBINARY"></a>
 *  \brief Methods of the class to set a feature binary categorical.
 */
class OneHotEncoder : public ModifierIface
{
    size_t idx;
    size_t nCats;

public:
    OneHotEncoder(size_t idx, size_t nCats) : idx(idx), nCats(nCats) {}

    virtual ~OneHotEncoder() {}

    virtual void apply(services::Collection<functionT> & funcList, services::Collection<FeatureAuxData> & auxVect) const override
    {
        size_t nCols = funcList.size();

        if (idx < nCols)
        {
            funcList[idx] = binFunc;
            auxVect[idx].buffer.resize(1024);
            auxVect[idx].nCats = nCats;
            auxVect[idx].wide  = nCats;
        }

        size_t nNTCols = 0;
        for (size_t i = 0; i < nCols; i++)
        {
            auxVect[i].idx = nNTCols;
            nNTCols += auxVect[i].wide;
        }
    }
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__COLUMNFILTER"></a>
 *  \brief Methods of the class to filter out data source features from output numeric table.
 */
class ColumnFilter : public ModifierIface
{
    bool oddFlag;
    bool evenFlag;
    bool noneFlag;
    bool listFlag;
    services::Collection<size_t> validList;

public:
    ColumnFilter() : oddFlag(false), evenFlag(false), noneFlag(false), listFlag(false) {}

    virtual ~ColumnFilter() {}

    ColumnFilter & odd()
    {
        oddFlag = true;
        return *this;
    }
    ColumnFilter & even()
    {
        evenFlag = true;
        return *this;
    }
    ColumnFilter & none()
    {
        noneFlag = true;
        return *this;
    }
    ColumnFilter & list(services::Collection<size_t> valid)
    {
        validList = valid;
        listFlag  = true;
        return *this;
    }

    virtual void apply(services::Collection<functionT> & funcList, services::Collection<FeatureAuxData> & auxVect) const override
    {
        size_t nCols = funcList.size();

        if (oddFlag)
        {
            for (size_t i = 0; i < nCols; i += 2)
            {
                funcList[i]     = nullFunc;
                auxVect[i].wide = 0;
            }
        }

        if (evenFlag)
        {
            for (size_t i = 1; i < nCols; i += 2)
            {
                funcList[i]     = nullFunc;
                auxVect[i].wide = 0;
            }
        }

        if (noneFlag)
        {
            for (size_t i = 0; i < nCols; i++)
            {
                funcList[i]     = nullFunc;
                auxVect[i].wide = 0;
            }
        }

        if (listFlag)
        {
            services::Collection<bool> flags(nCols);

            for (size_t i = 0; i < nCols; i++)
            {
                flags[i] = false;
            }

            for (size_t i = 0; i < validList.size(); i++)
            {
                size_t el = validList[i];
                if (el < nCols)
                {
                    flags[el] = true;
                }
            }

            for (size_t i = 0; i < nCols; i++)
            {
                if (flags[i]) continue;
                funcList[i]     = nullFunc;
                auxVect[i].wide = 0;
            }
        }

        size_t nNTCols = 0;
        for (size_t i = 0; i < nCols; i++)
        {
            auxVect[i].idx = nNTCols;
            nNTCols += auxVect[i].wide;
        }
    }
};

namespace interface1
{
/**
 * @defgroup data_sources Data Sources
 * \brief Specifies methods to access data
 * @ingroup data_management
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__CSVFEATUREMANAGER"></a>
 *  \brief Methods of the class to preprocess data represented in the CSV format.
 */
class CSVFeatureManager : public StringRowFeatureManagerIface
{
public:
    /**
     *  Default constructor
     */
    CSVFeatureManager() : _delimiter(','), _numberOfTokens(0) {}

    virtual ~CSVFeatureManager() {}

    /**
     *  Sets a new character as a delimiter for parsing CSV data (default ',')
     */
    void setDelimiter(char delimiter) { _delimiter = delimiter; }

public:
    /**
     * Gets number of columns which must be allocated in numeric table
     * \return The number of columns in numeric table
     */
    size_t getNumericTableNumberOfColumns() const
    {
        if (_modifiersManager)
        {
            return _modifiersManager->getNumberOfOutputFeatures();
        }

        const size_t nDSCols = auxVect.size();
        return auxVect[nDSCols - 1].idx + auxVect[nDSCols - 1].wide;
    }

    /**
     * Sets information about features from the given dictionary
     * \param dictionary The data source dictionary
     */
    services::Status setFeatureDetailsFromDictionary(DataSourceDictionary * dictionary)
    {
        DAAL_CHECK(dictionary, services::ErrorNullPtr);

        auxVect.clear();
        funcList.clear();
        fillAuxVectAndFuncList(*dictionary);
        _numberOfTokens = dictionary->getNumberOfFeatures();

        return services::Status();
    }

    /**
     * Adds a simple feature modifier
     * \param[in]  modifier The modifier
     */
    void addModifier(const ModifierIface & modifier) { modifier.apply(funcList, auxVect); }

    /**
     * Adds extended feature modifier
     * \param[in]   featureIds The identifiers of the features to be modified
     * \param[in]   modifier   The feature modifier
     * \param[out]  status     (optional) The pointer to status object
     * \return Reference to itself
     */
    CSVFeatureManager & addModifier(const features::FeatureIdCollectionIfacePtr & featureIds,
                                    const modifiers::csv::FeatureModifierIfacePtr & modifier, services::Status * status = NULL)
    {
        services::Status localStatus;
        if (!_modifiersManager)
        {
            _modifiersManager = modifiers::csv::internal::ModifiersManager::create(&localStatus);
            if (!localStatus)
            {
                services::internal::tryAssignStatusAndThrow(status, localStatus);
                return *this;
            }
        }

        localStatus |= _modifiersManager->addModifier(featureIds, modifier);
        if (!localStatus)
        {
            services::internal::tryAssignStatusAndThrow(status, localStatus);
            return *this;
        }

        return *this;
    }

    /**
     * Parses a string that represents header of CSV data
     * \param[in]  rawRowData   Array of characters with the string that represents the feature vector
     * \param[in]  rawDataSize  Size of the rawRowData array
     */
    void parseRowAsHeader(char * rawRowData, size_t rawDataSize)
    {
        DAAL_ASSERT(rawRowData);

        internal::CSVRowTokenizer tokenizer(rawRowData, rawDataSize, _delimiter);
        for (tokenizer.reset(); tokenizer.good(); tokenizer.next())
        {
            _featuresInfo.addFeatureName(tokenizer.getCurrentToken());
        }
    }

    /**
     * Parses a string that represents a feature vector and fills provided data source dictionary
     * \param[in]  rawRowData   Array of characters with the string that represents the feature vector
     * \param[in]  rawDataSize  Size of the rawRowData array
     * \param[in]  dictionary   Pointer to the dictionary
     */
    virtual void parseRowAsDictionary(char * rawRowData, size_t rawDataSize, DataSourceDictionary * dictionary) override
    {
        DAAL_ASSERT(rawRowData);
        DAAL_ASSERT(dictionary);

        _numberOfTokens = 0;

        internal::CSVRowTokenizer tokenizer(rawRowData, rawDataSize, _delimiter);
        for (tokenizer.reset(); tokenizer.good(); tokenizer.next())
        {
            _numberOfTokens++;
            _featuresInfo.addFeatureType(tokenizer.getCurrentToken());
        }

        if (_modifiersManager)
        {
            _modifiersManager->prepare(_featuresInfo);
            _modifiersManager->fillDictionary(*dictionary);
        }
        else
        {
            fillDictionaryWithoutModifiers(*dictionary);
        }
    }

    /**
     *  Parses a string that represents a feature vector and converts it into a numeric representation
     *  \param[in]  rawRowData   Array of characters with the string that represents the feature vector
     *  \param[in]  rawDataSize  Size of the rawRowData array
     *  \param[in]  dictionary   Pointer to the dictionary
     *  \param[out] rowBuffer    Pointer to a Buffer View to store the result of parsing
     *  \param[in]  ntRowIndex   Position in the Numeric Table at which to store the result of parsing
     */
    virtual void parseRowIn(char * rawRowData, size_t rawDataSize, DataSourceDictionary * DAAL_ASSERT_DECL(dictionary),
                            services::BufferView<DAAL_DATA_TYPE> & rowBuffer, size_t /*ntRowIndex*/) override
    {
        DAAL_ASSERT(dictionary);
        DAAL_ASSERT(rawRowData);

        size_t i = 0;
        internal::CSVRowTokenizer tokenizer(rawRowData, rawDataSize, _delimiter);

        if (_modifiersManager)
        {
            for (tokenizer.reset(); tokenizer.good() && i < _numberOfTokens; tokenizer.next(), i++)
            {
                _modifiersManager->setToken(i, tokenizer.getCurrentToken());
            }
            _modifiersManager->applyModifiers(rowBuffer);
        }
        else
        {
            DAAL_DATA_TYPE * row = rowBuffer.data();
            for (tokenizer.reset(); tokenizer.good() && i < _numberOfTokens; tokenizer.next(), i++)
            {
                const services::StringView token = tokenizer.getCurrentToken();
                funcList[i](token.c_str(), auxVect[i], row);
            }
        }
    }

    /**
     * Finalizes CSV data parsing
     * \param[in]  dictionary  Pointer to the dictionary
     */
    void finalize(DataSourceDictionary * dictionary)
    {
        if (_modifiersManager)
        {
            _modifiersManager->finalize();
            _modifiersManager->fillDictionary(*dictionary);
        }
    }

private:
    void fillDictionaryWithoutModifiers(DataSourceDictionary & dictionary)
    {
        const size_t nFeatures = _featuresInfo.getNumberOfFeatures();
        dictionary.setNumberOfFeatures(nFeatures);

        for (size_t i = 0; i < nFeatures; i++)
        {
            features::FeatureType fType         = _featuresInfo.getDetectedFeatureType(i);
            dictionary[i].ntFeature.featureType = fType;

            switch (fType)
            {
            case features::DAAL_CONTINUOUS: dictionary[i].ntFeature.setType<DAAL_DATA_TYPE>(); break;

            case features::DAAL_ORDINAL:
            case features::DAAL_CATEGORICAL: dictionary[i].ntFeature.setType<int>(); break;
            }
        }

        fillAuxVectAndFuncList(dictionary);
    }

    void fillAuxVectAndFuncList(DataSourceDictionary & dictionary)
    {
        const size_t nFeatures = dictionary.getNumberOfFeatures();
        auxVect.resize(nFeatures);
        funcList.resize(nFeatures);

        for (size_t i = 0; i < nFeatures; i++)
        {
            DataSourceFeature & feature     = dictionary[i];
            NumericTableFeature & ntFeature = feature.ntFeature;

            auxVect.push_back(FeatureAuxData(i, &feature, &ntFeature));
            funcList.push_back(getModifierFunctionPtr(ntFeature));
        }
    }

    static functionT getModifierFunctionPtr(const NumericTableFeature & ntFeature)
    {
        switch (ntFeature.featureType)
        {
        case features::DAAL_CONTINUOUS: return ModifierIface::contFunc;

        case features::DAAL_ORDINAL:
        case features::DAAL_CATEGORICAL: return ModifierIface::catFunc;
        }
        return ModifierIface::nullFunc;
    }

protected:
    char _delimiter;
    services::Collection<functionT> funcList;
    services::Collection<FeatureAuxData> auxVect;

private:
    size_t _numberOfTokens;
    BlockDescriptor<DAAL_DATA_TYPE> _currentRowBlock;

    internal::CSVFeaturesInfo _featuresInfo;
    modifiers::csv::internal::ModifiersManagerPtr _modifiersManager;
};
/** @} */
} // namespace interface1

using interface1::CSVFeatureManager;

} // namespace data_management
} // namespace daal

#endif
