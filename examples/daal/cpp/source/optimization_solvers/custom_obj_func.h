/* file: custom_obj_func.h */
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
!  Content:
!    Interface and implementation of user-defined algorithm for computation of
!    logistic loss function in DAAL style
!
!******************************************************************************/

#include <vector>
#include <math.h>
#include <stdint.h>

#include "daal.h"

/* namespace for the new loss function algorithm */
namespace new_objective_function {
/**
 * Available identifiers of input objects of the logistic loss objective function
 */
enum InputId {
    argument = (int)daal::algorithms::optimization_solver::sum_of_functions::
        argument, /*!< Numeric table of size 1 x p with input argument of the objective function */
    data, /*!< Numeric table of size n x p with data */
    dependentVariables, /*!< Numeric table of size n x 1 with dependent variables */
    lastInputId = dependentVariables
};

/**
 * \brief %Parameter for logistic loss error objective function
 */
struct Parameter : public daal::algorithms::optimization_solver::sum_of_functions::Parameter {
    typedef daal::algorithms::optimization_solver::sum_of_functions::Parameter super;
    /**
     * Constructs the parameter of logistic loss objective function
     * \param[in] numberOfTerms    The number of terms in the function
     * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
     *                             a batch of indices used to compute the function results, e.g.,
     *                             value of the sum of the functions. If no indices are provided,
     *                             all terms will be used in the computations.
     * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
     */
    Parameter(size_t numberOfTerms,
              daal::data_management::NumericTablePtr batchIndices =
                  daal::data_management::NumericTablePtr(),
              const uint64_t resultsToCompute =
                  daal::algorithms::optimization_solver::objective_function::gradient)
            : super(numberOfTerms, batchIndices, resultsToCompute) {}

    /**
     * Constructs an Parameter by copying input objects and parameters of another Parameter
     * \param[in] other An object to be used as the source to initialize object
     */
    Parameter(const Parameter &other) : super(other) {}

    /**
     * Checks the correctness of the parameter
     * \return Status of computations
     */
    virtual daal::services::Status check() const override {
        return super::check();
    }
};

/**
 * \brief %Input objects for the logistic loss objective function
 */
class Input : public daal::algorithms::optimization_solver::sum_of_functions::Input {
public:
    typedef daal::algorithms::optimization_solver::sum_of_functions::Input super;

    /** Default constructor */
    Input() : super(lastInputId + 1) {}

    /** Copy constructor */
    Input(const Input &other) : super(other) {}

    /**
     * Sets one input object for logistic loss objective function
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const daal::data_management::NumericTablePtr &ptr) {
        Argument::set(id, ptr);
    }

    /**
     * Returns the input numeric table for logistic loss objective function
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input object that corresponds to the given identifier
     */
    daal::data_management::NumericTablePtr get(InputId id) const {
        return daal::data_management::NumericTable::cast(Argument::get(id));
    }

    /**
     * Checks the correctness of the input
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     *
     * \return Status of computations
     */
    daal::services::Status check(const daal::algorithms::Parameter *par,
                                 int method) const override {
        using namespace daal::data_management;

        super::check(par, method);

        NumericTablePtr xTable = get(data);
        daal::services::Status s = checkNumericTable(xTable.get(), "data", 0, 0);
        if (!s)
            return s;

        const size_t nColsInData = xTable->getNumberOfColumns();
        const size_t nRowsInData = xTable->getNumberOfRows();

        s = checkNumericTable(get(dependentVariables).get(),
                              "dependentVariables",
                              0,
                              0,
                              1,
                              nRowsInData);
        s |= checkNumericTable(get(argument).get(), "argument", 0, 0, 1, nColsInData + 1);
        return s;
    }
};

/**
 * \brief Provides methods to run implementations of the logistic loss objective function.
 *        This class is associated with the Batch class and supports the method of computing
 *        the logistic loss objective function in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the logistic loss objective function, double or float
 */
template <typename algorithmFPType>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<daal::batch> {
public:
    /**
     * Constructs a container for the logistic loss objective function in the batch processing mode
     */
    BatchContainer(daal::services::Environment::env * /*daalEnv*/) {}

    /**
     * Computes the result of the logistic loss objective function in the batch processing mode
     * \return Status of computations
     */
    virtual daal::services::Status compute() override;
};

/**
 * \brief Computes the logistic loss objective function in the batch processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the logistic loss objective function, double or float
 */
template <typename algorithmFPType = float>
class Batch : public daal::algorithms::optimization_solver::sum_of_functions::Batch {
public:
    typedef daal::algorithms::optimization_solver::sum_of_functions::Batch super;

    Input input; /*!< %Input data structure */
    Parameter parameter; /*!< %Parameter data structure */

    /**
     * Main constructor
     */
    Batch(size_t numberOfTerms)
            : super(numberOfTerms, &input, &parameter),
              parameter(numberOfTerms) {
        initialize();
    }

    /**
     * Constructs an the Mean squared error objective function algorithm by copying input objects and parameters
     * of another the Mean squared error objective function algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType> &other)
            : super(other.parameter.numberOfTerms, &input, &parameter),
              input(other.input),
              parameter(other.parameter) {
        initialize();
    }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const override {
        return 0;
    }

    /**
     * Returns a pointer to the newly allocated the Mean squared error objective function algorithm with a copy of input objects
     * of this the Mean squared error objective function algorithm
     * \return Pointer to the newly allocated algorithm
     */
    daal::services::SharedPtr<Batch<algorithmFPType> > clone() const {
        return daal::services::SharedPtr<Batch<algorithmFPType> >(cloneImpl());
    }

    /**
     * Allocates memory buffers needed for the computations
     * \return Status of computations
     */
    daal::services::Status allocate() {
        return allocateResult();
    }

protected:
    virtual Batch<algorithmFPType> *cloneImpl() const override {
        return new Batch<algorithmFPType>(*this);
    }

    virtual daal::services::Status allocateResult() override {
        daal::services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, 0);
        _res = _result.get();
        return s;
    }

    void initialize() {
        daal::algorithms::Analysis<daal::batch>::_ac = new BatchContainer<algorithmFPType>(&_env);
        _in = &input;
        _par = &parameter;
    }
};

// implementation of the algorithm for computation of the logistic loss objective function
template <typename algorithmFPType>
daal::services::Status BatchContainer<algorithmFPType>::compute() {
    using namespace daal::data_management;

    Input *input = static_cast<Input *>(_in);
    daal::algorithms::optimization_solver::objective_function::Result *result =
        static_cast<daal::algorithms::optimization_solver::objective_function::Result *>(_res);
    Parameter *parameter = static_cast<Parameter *>(_par);

    NumericTable *xTable = input->get(data).get(); // input data set
    NumericTable *yTable = input->get(dependentVariables).get(); // array of dependent variables
    NumericTable *argumentTable = input->get(argument).get(); // argument of the objective function
    NumericTable *indicesTable = parameter->batchIndices.get(); // stochastic indices

    const size_t p = argumentTable->getNumberOfRows(); // size of the argument
    const size_t dim = p - 1; // number of features in the input data set
    const algorithmFPType one(1.0);

    BlockDescriptor<int> indicesBlock;
    indicesTable->getBlockOfRows(0, 1, readOnly, indicesBlock);
    const int *indices = indicesBlock.getBlockPtr();
    const size_t nIndices = indicesTable->getNumberOfColumns();
    const algorithmFPType invN = one / (algorithmFPType)nIndices;

    std::vector<algorithmFPType> x(nIndices * dim);
    std::vector<algorithmFPType> y(nIndices);

    /* Get the subset of the rows from the data set and the dependent variables */
    for (size_t i = 0; i < nIndices; i++) {
        /* Get a row of data from the input data set */
        BlockDescriptor<algorithmFPType> xBlock;
        xTable->getBlockOfRows(indices[i], 1, readOnly, xBlock);
        algorithmFPType *xRow = xBlock.getBlockPtr();

        for (size_t j = 0; j < dim; j++)
            x[i * dim + j] = xRow[j];

        xTable->releaseBlockOfRows(xBlock);

        /* Get a dependent variable */
        BlockDescriptor<algorithmFPType> yBlock;
        yTable->getBlockOfRows(indices[i], 1, readOnly, yBlock);
        algorithmFPType *yVal = yBlock.getBlockPtr();

        y[i] = *yVal;

        yTable->releaseBlockOfRows(yBlock);
    }
    indicesTable->releaseBlockOfRows(indicesBlock);

    std::vector<algorithmFPType> f(nIndices);
    std::vector<algorithmFPType> s(nIndices);

    /* Get data as an array from the dependent variables */
    BlockDescriptor<algorithmFPType> argumentBlock;
    argumentTable->getBlockOfRows(0, 1, readOnly, argumentBlock);
    const algorithmFPType *argumentArray = argumentBlock.getBlockPtr();
    const algorithmFPType theta0 = argumentArray[0];
    const algorithmFPType *theta = &argumentArray[1];

    for (size_t i = 0; i < nIndices; i++) {
        f[i] = theta0;
        for (size_t j = 0; j < dim; j++) {
            f[i] += theta[j] * x[i * dim + j];
        }
        s[i] = one / (one + exp(-f[i]));
    }
    argumentTable->releaseBlockOfRows(argumentBlock);

    /* Compute value of the logistic loss function */
    const bool valueFlag = ((parameter->resultsToCompute &
                             daal::algorithms::optimization_solver::objective_function::value) != 0)
                               ? true
                               : false;
    if (valueFlag) {
        NumericTable *valueTable =
            result->get(daal::algorithms::optimization_solver::objective_function::valueIdx).get();
        BlockDescriptor<algorithmFPType> valueBlock;
        valueTable->getBlockOfRows(0, 1, writeOnly, valueBlock);
        algorithmFPType *value = valueBlock.getBlockPtr();
        value[0] = 0.0;
        for (size_t i = 0; i < nIndices; i++) {
            value[0] += y[i] * log(s[i]) + (one - y[i]) * log(one - s[i]);
        }
        value[0] *= -invN;
        valueTable->releaseBlockOfRows(valueBlock);
    }

    /* Compute gradient of the logistic loss function */
    const bool gradientFlag =
        ((parameter->resultsToCompute &
          daal::algorithms::optimization_solver::objective_function::gradient) != 0)
            ? true
            : false;
    if (gradientFlag) {
        NumericTable *gradientTable =
            result->get(daal::algorithms::optimization_solver::objective_function::gradientIdx)
                .get();
        BlockDescriptor<algorithmFPType> gradientBlock;
        gradientTable->getBlockOfRows(0, p, writeOnly, gradientBlock);
        algorithmFPType *gradient = gradientBlock.getBlockPtr();

        for (size_t j = 0; j < p; j++) {
            gradient[j] = 0.0;
        }
        for (size_t i = 0; i < nIndices; i++) {
            gradient[0] += (s[i] - y[i]);
            for (size_t j = 0; j < dim; j++) {
                gradient[j + 1] += (s[i] - y[i]) * x[i * dim + j];
            }
        }

        for (size_t j = 0; j < p; j++) {
            gradient[j] *= invN;
        }
        gradientTable->releaseBlockOfRows(gradientBlock);
    }

    /* Compute Hessian of the logistic loss function */
    const bool hessianFlag =
        ((parameter->resultsToCompute &
          daal::algorithms::optimization_solver::objective_function::hessian) != 0)
            ? true
            : false;
    if (hessianFlag) {
        NumericTable *hessianTable =
            result->get(daal::algorithms::optimization_solver::objective_function::hessianIdx)
                .get();
        /* Hessian computations to go here */

        delete hessianTable;
    }

    return daal::services::Status();
}
} // namespace new_objective_function
