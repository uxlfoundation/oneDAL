/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "daal/include/data_management/data/data_archive.h"
#include "oneapi/dal/detail/serialization.hpp"

namespace oneapi::dal::backend::interop {

class daal_data_archive_stub : public base, public daal::data_management::DataArchiveIface {
public:
    daal::services::SharedPtr<daal::byte> getArchiveAsArraySharedPtr() const override {
        ONEDAL_ASSERT(!"Not implemented");
        return daal::services::SharedPtr<daal::byte>{};
    }

    daal::byte* getArchiveAsArray() override {
        ONEDAL_ASSERT(!"Not implemented");
        return nullptr;
    }

    std::string getArchiveAsString() override {
        ONEDAL_ASSERT(!"Not implemented");
        return std::string{};
    }

    std::size_t copyArchiveToArray(daal::byte* ptr, std::size_t maxLength) const override {
        ONEDAL_ASSERT(!"Not implemented");
        return 0;
    }

    void setMajorVersion(int majorVersion) override {
        major_version_ = majorVersion;
    }

    void setMinorVersion(int minorVersion) override {
        minor_version_ = minorVersion;
    }

    void setUpdateVersion(int updateVersion) override {
        update_version_ = updateVersion;
    }

    int getMajorVersion() override {
        return major_version_;
    }

    int getMinorVersion() override {
        return minor_version_;
    }

    int getUpdateVersion() override {
        return update_version_;
    }

private:
    int major_version_;
    int minor_version_;
    int update_version_;
};

class daal_input_data_archive_adapter : public daal_data_archive_stub {
public:
    explicit daal_input_data_archive_adapter(detail::input_archive& archive)
            : archive_(archive),
              size_counter_(0) {}

    void write(daal::byte* ptr, std::size_t size) override {
        ONEDAL_ASSERT(!"Write is prohibited for input archive");
    }

    void read(daal::byte* ptr, std::size_t size) override {
        size_counter_ += size;
        archive_.range(ptr, ptr + size);
    }

    std::size_t getSizeOfArchive() const override {
        return size_counter_;
    }

private:
    detail::input_archive& archive_;
    std::size_t size_counter_;
};

class daal_output_data_archive_adapter : public daal_data_archive_stub {
public:
    explicit daal_output_data_archive_adapter(detail::output_archive& archive)
            : archive_(archive),
              size_counter_(0) {}

    void write(daal::byte* ptr, std::size_t size) override {
        size_counter_ += size;
        archive_.range(ptr, ptr + size);
    }

    void read(daal::byte* ptr, std::size_t size) override {
        ONEDAL_ASSERT(!"Read is prohibited for output archive");
    }

    std::size_t getSizeOfArchive() const override {
        return size_counter_;
    }

private:
    detail::output_archive& archive_;
    std::size_t size_counter_;
};

class daal_input_data_archive : public daal::data_management::OutputDataArchive {
    using base_t = daal::data_management::OutputDataArchive;

public:
    explicit daal_input_data_archive(detail::input_archive& archive)
            : base_t(new daal_input_data_archive_adapter{ archive }) {}

    daal_input_data_archive(const daal_input_data_archive&) = delete;
    daal_input_data_archive& operator=(const daal_input_data_archive&) = delete;
    ~daal_input_data_archive() override = default;
};

class daal_output_data_archive : public daal::data_management::InputDataArchive {
    using base_t = daal::data_management::InputDataArchive;

public:
    explicit daal_output_data_archive(detail::output_archive& archive)
            : base_t(new daal_output_data_archive_adapter{ archive }) {}

    daal_output_data_archive(const daal_output_data_archive&) = delete;
    daal_output_data_archive& operator=(const daal_output_data_archive&) = delete;
    ~daal_output_data_archive() override = default;
};

} // namespace oneapi::dal::backend::interop
