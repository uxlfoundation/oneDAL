/* file: NumericTableInputId.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

/**
 * @ingroup prediction
 * @{
 */
package com.intel.daal.algorithms.classifier.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__NUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input objects of classification algorithms
 */
public final class NumericTableInputId {
    private int _value;

    /**
     * Constructs the numeric table input object identifier using the provided value
     * @param value     Value corresponding to the numeric table input object identifier
     */
    public NumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the numeric table input object identifier
     * @return Value corresponding to the numeric table input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int Data = 0;

    public static final NumericTableInputId data = new NumericTableInputId(Data); /*!< Data for the prediction stage*/

    public static boolean validate(NumericTableInputId id) {
        return id.getValue() == data.getValue();
    }

    public static void throwIfInvalid(NumericTableInputId id) {
        if (id == null) {
            throw new IllegalArgumentException("Null input id");
        }
        if (!NumericTableInputId.validate(id)) {
            throw new IllegalArgumentException("Unsupported input id");
        }
    }
}
/** @} */
