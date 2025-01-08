/** file algorithm_base_impl.cpp */
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
//
//--
*/

#include "algorithms/algorithm_base.h"
#include "algorithms/algorithm_base_mode_impl.h"
#include "src/algorithms/argument_storage.h"
#include "src/services/service_algo_utils.h"

#include "src/threading/service_thread_pinner.h"
#include "src/services/service_topo.h"

namespace daal
{
namespace algorithms
{
#if !(defined DAAL_THREAD_PINNING_DISABLED)
template <typename AlgorithmType>
class TaskWrapper : public services::internal::thread_pinner_task_t
{
    AlgorithmType * _alg;
    services::Status _status;

public:
    TaskWrapper(AlgorithmType * alg) : _alg(alg) {}

    virtual void operator()() { _status |= _alg->compute(); }

    const services::Status & getStatus() const { return _status; }
};
#endif

algorithms::Argument::Argument(const size_t n) : _storage(new internal::ArgumentStorage(n)), idx(0) {}

algorithms::Argument::Argument(const algorithms::Argument & other)
    : _storage(new internal::ArgumentStorage(*(internal::ArgumentStorage *)other._storage.get())), idx(0)
{}

const data_management::SerializationIfacePtr & algorithms::Argument::get(size_t index) const
{
    return (*_storage)[index];
}

void algorithms::Argument::set(size_t index, const data_management::SerializationIfacePtr & value)
{
    (*_storage)[index] = value;
}

void algorithms::Argument::setStorage(const data_management::DataCollectionPtr & storage)
{
    _storage = storage;
}

data_management::DataCollectionPtr & algorithms::Argument::getStorage(Argument & a)
{
    return a._storage;
}

const data_management::DataCollectionPtr & algorithms::Argument::getStorage(const Argument & a)
{
    return a._storage;
}

services::SharedPtr<Base> internal::ArgumentStorage::getExtension(Extension type)
{
    services::SharedPtr<Base> ptr;
    if (int(type) < _extensions.size()) ptr = _extensions[type];
    return ptr;
}

void internal::ArgumentStorage::setExtension(Extension type, const services::SharedPtr<Base> & ptr)
{
    if (int(type) >= _extensions.size())
    {
        for (auto i = _extensions.size(); i < type; ++i) _extensions.push_back(services::SharedPtr<Base>());
        _extensions.push_back(ptr);
    }
    else
    {
        _extensions[type] = ptr;
    }
}
} // namespace algorithms

namespace services
{
namespace internal
{
services::HostAppIfacePtr getHostApp(algorithms::internal::ArgumentStorage & s)
{
    auto ext = s.getExtension(algorithms::internal::ArgumentStorage::hostApp);
    DAAL_ASSERT(!ext.get() || dynamic_cast<services::HostAppIface *>(ext.get()));
    return services::dynamicPointerCast<services::HostAppIface>(ext);
}

//service class that makes possible to access Input's storage
class StorageAccessor : public daal::algorithms::Input
{
public:
    static daal::algorithms::internal::ArgumentStorage * get(daal::algorithms::Input & inp)
    {
        return dynamic_cast<daal::algorithms::internal::ArgumentStorage *>(getStorage(inp).get());
    }
};

services::HostAppIfacePtr getHostApp(daal::algorithms::Input & inp)
{
    auto storage = StorageAccessor::get(inp);
    if (storage) return internal::getHostApp(*storage);
    return services::HostAppIfacePtr();
}

services::HostAppIface * hostApp(daal::algorithms::Input & inp)
{
    auto storage = StorageAccessor::get(inp);
    return storage ? getHostApp(*storage).get() : nullptr;
}

void setHostApp(const services::SharedPtr<services::HostAppIface> & pHostApp, daal::algorithms::Input & inp)
{
    auto ptr = StorageAccessor::get(inp);
    if (ptr) ptr->setExtension(algorithms::internal::ArgumentStorage::hostApp, pHostApp);
}

} //namespace internal
} //namespace services

namespace algorithms
{
namespace interface1
{

AlgorithmContainerIfaceImpl::AlgorithmContainerIfaceImpl() = default;
AlgorithmContainer<batch>::AlgorithmContainer()            = default;

template <ComputeMode mode>
AlgorithmContainer<mode>::AlgorithmContainer() = default;

} //namespace interface1

template <ComputeMode mode>
services::Status AlgorithmImpl<mode>::computeNoThrow()
{
    this->setParameter();

    services::Status s;
    if (this->isChecksEnabled())
    {
        s = this->checkComputeParams();
        if (!s) return s;
    }

    DAAL_CHECK_MALLOC(this->allocatePartialResultMemory());

    this->_ac->setArguments(this->_in, this->_pres, this->_par, this->_hpar);

    if (this->isChecksEnabled())
    {
        s = this->checkResult();
        if (!s) return s;
    }

    if (!this->getInitFlag())
    {
        s = this->initPartialResult();
        if (!s) return s;
        this->setInitFlag(true);
    }

    s = setupCompute();
    if (s)
    {
#if !(defined DAAL_THREAD_PINNING_DISABLED)
        daal::services::internal::thread_pinner_t * pinner = daal::services::internal::getThreadPinner(false, read_topology, delete_topology);

        if (pinner != NULL)
        {
            TaskWrapper<AlgorithmContainerImpl<mode> > task(this->_ac);
            pinner->execute(task);
            s |= task.getStatus();
        }
        else
#endif
        {
            s = this->_ac->compute();
        }
    }

    s |= resetCompute();
    return s;
}

template <ComputeMode mode>
services::HostAppIfacePtr AlgorithmImpl<mode>::hostApp()
{
    return this->_in ? services::internal::getHostApp(*this->_in) : services::HostAppIfacePtr();
}

template <ComputeMode mode>
void AlgorithmImpl<mode>::setHostApp(const services::HostAppIfacePtr & pHost)
{
    if (this->_in) services::internal::setHostApp(pHost, *this->_in);
}

/**
 * Computes final results of the algorithm in the %batch mode without possibility of throwing an exception.
 */
services::Status AlgorithmImpl<batch>::computeNoThrow()
{
    this->setParameter();

    if (this->isChecksEnabled())
    {
        services::Status _s = this->checkComputeParams();
        if (!_s) return _s;
    }

    services::Status s = this->allocateResultMemory();
    DAAL_CHECK_MALLOC(s);

    this->_ac->setArguments(this->_in, this->_res, this->_par, this->_hpar);

    if (this->isChecksEnabled())
    {
        s = this->checkResult();
        if (!s) return s;
    }
    s = setupCompute();

    if (s)
    {
#if !(defined DAAL_THREAD_PINNING_DISABLED)
        daal::services::internal::thread_pinner_t * pinner = daal::services::internal::getThreadPinner(false, read_topology, delete_topology);

        if (pinner != NULL)
        {
            TaskWrapper<AlgorithmContainerImpl<batch> > task(_ac);
            pinner->execute(task);
            s |= task.getStatus();
        }
        else
#endif
        {
            s |= this->_ac->compute();
        }
    }

    if (resetFlag) s |= resetCompute();
    _res = this->_ac->getResult();
    return s;
}

services::HostAppIfacePtr AlgorithmImpl<batch>::hostApp()
{
    return this->_in ? services::internal::getHostApp(*this->_in) : services::HostAppIfacePtr();
}

void AlgorithmImpl<batch>::setHostApp(const services::HostAppIfacePtr & pHost)
{
    if (this->_in) services::internal::setHostApp(pHost, *this->_in);
}

template class interface1::AlgorithmImpl<online>;
template class interface1::AlgorithmImpl<distributed>;
} // namespace algorithms
} // namespace daal
