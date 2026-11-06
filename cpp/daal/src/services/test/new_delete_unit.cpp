/*******************************************************************************
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

#include <new>
#include <stdexcept>
#include <type_traits>

#include "oneapi/dal/test/engine/common.hpp"
#include "services/base.h"
#include "services/daal_shared_ptr.h"

namespace daal::test
{
class ThrowsOnConstruction
{
public:
    DAAL_NEW_DELETE();

    ThrowsOnConstruction() { throw std::runtime_error("constructor failed"); }
};

class DerivedFromBase : public Base
{};

using PlacementDelete = void (*)(void *, void *);

// MSVC warns with C4291 on every placement new of a class that lacks these, e.g. in Cython's
// construction of SharedPtr members.
static_assert(std::is_same_v<decltype(static_cast<PlacementDelete>(&services::SharedPtr<int>::operator delete)), PlacementDelete>);
static_assert(std::is_same_v<decltype(static_cast<PlacementDelete>(&services::SharedPtr<int>::operator delete[])), PlacementDelete>);
static_assert(std::is_same_v<decltype(static_cast<PlacementDelete>(&DerivedFromBase::operator delete)), PlacementDelete>);
static_assert(std::is_same_v<decltype(static_cast<PlacementDelete>(&DerivedFromBase::operator delete[])), PlacementDelete>);

TEST("a throwing constructor in placement new leaves the caller's storage alone", "[services][unit]")
{
    alignas(ThrowsOnConstruction) unsigned char storage[sizeof(ThrowsOnConstruction)];
    REQUIRE_THROWS_AS(new (storage) ThrowsOnConstruction(), std::runtime_error);
}

TEST("placement new constructs a SharedPtr in caller-owned storage", "[services][unit]")
{
    using Ptr = services::SharedPtr<int>;
    alignas(Ptr) unsigned char storage[sizeof(Ptr)];
    Ptr * const ptr = new (storage) Ptr();
    REQUIRE(ptr->get() == nullptr);
    ptr->~Ptr();
}

TEST("new and delete of a Base subclass still go through the DAAL allocator", "[services][unit]")
{
    DerivedFromBase * const single = new DerivedFromBase();
    REQUIRE(single != nullptr);
    delete single;

    DerivedFromBase * const array = new DerivedFromBase[4];
    REQUIRE(array != nullptr);
    delete[] array;
}

} // namespace daal::test
