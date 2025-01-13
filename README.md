<!--
******************************************************************************
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
*******************************************************************************/-->

# oneAPI Data Analytics Library <!-- omit in toc --> <img align="right" width="200" height="100" src="https://raw.githubusercontent.com/uxlfoundation/artwork/e98f1a7a3d305c582d02c5f532e41487b710d470/foundation/uxl-foundation-logo-horizontal-color.svg">

[Installation](#installation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](#documentation)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Support](#support)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](#examples)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[How to Contribute](CONTRIBUTING.md)&nbsp;&nbsp;&nbsp;

[![Build Status](https://dev.azure.com/daal/DAAL/_apis/build/status/CI?repoName=uxlfoundation/oneDAL&branchName=main)](https://dev.azure.com/daal/DAAL/_build/latest?definitionId=7&branchName=main)
[![License](https://img.shields.io/github/license/uxlfoundation/oneDAL.svg)](https://github.com/uxlfoundation/oneDAL/blob/main/LICENSE)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8859/badge)](https://www.bestpractices.dev/projects/8859)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/uxlfoundation/oneDAL/badge)](https://securityscorecards.dev/viewer/?uri=github.com/uxlfoundation/oneDAL)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/uxlfoundation/oneDAL/discussions)

oneAPI Data Analytics Library (oneDAL) is a powerful machine learning library that helps you accelerate big data analysis at all stages: **preprocessing**, **transformation**, **analysis**, **modeling**, **validation**, and **decision making**.

The library implements classical machine learning algorithms. The boost in their performance is achieved by leveraging the capabilities of Intel&reg; hardware.

The oneDAL is part of the [UXL Foundation](http://www.uxlfoundation.org) and is an implementation of the [oneAPI specification](https://spec.oneapi.io) for oneDAL component.

## Usage

There are different ways for you to build high-performance data science applications that use the advantages of oneDAL:
- Use oneDAL C++ interfaces with or without SYCL support ([learn more](https://uxlfoundation.github.io/oneDAL/#oneapi-vs-daal-interfaces))
- Use [Intel(R) Extension for Scikit-learn*](https://uxlfoundation.github.io/scikit-learn-intelex/) to accelerate existing scikit-learn code without changing it
- Use [daal4py](https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/daal4py), a standalone package with Python API for oneDAL
Deprecation Notice: The Java interfaces are removed from the oneDAL library.


## Installation

Check the [System Requirements](https://uxlfoundation.github.io/oneDAL/system-requirements.html) before installing to ensure compatibility with your system.

There are several options available for installing oneDAL:

- **Binary Distribution**: You can download pre-built binary packages from the following sources:
    - Intel® oneAPI:
        - Download as Part of the [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html#gs.8xrue2)
        - Download as the Stand-Alone [Intel® oneAPI Data Analytics Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html#gs.8xrue2)
    - Anaconda:
        | Channel | Version |
        |:-------:|:-------:|
        | intel | [![Anaconda-Server Intel Badge](https://anaconda.org/intel/dal-devel/badges/version.svg)](https://anaconda.org/intel/dal-devel) |
        | conda-forge | [![Anaconda-Server Conda-forge Badge](https://anaconda.org/conda-forge/dal-devel/badges/version.svg)](https://anaconda.org/conda-forge/dal-devel) |

    - [NuGet](https://www.nuget.org/packages/inteldal.devel.linux-x64)

- **Source Distribution**: You can build the library from source. To do this, [download the specific version of oneDAL](https://github.com/uxlfoundation/oneDAL/releases) from the official GitHub repository and follow the instructions in the [INSTALL.md](INSTALL.md).


## Examples

C++ Examples:

- [oneAPI interfaces with SYCL support](https://github.com/uxlfoundation/oneDAL/tree/main/examples/oneapi/dpc)
- [oneAPI interfaces without SYCL support](https://github.com/uxlfoundation/oneDAL/tree/main/examples/oneapi/cpp)
- [DAAL interfaces](https://github.com/uxlfoundation/oneDAL/tree/main/examples/daal/cpp)

Python Examples:
- [scikit-learn-intelex](https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/notebooks)
- [daal4py](https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py)

<details><summary>Other Examples</summary>

- [MPI](https://github.com/uxlfoundation/oneDAL/tree/main/samples/daal/cpp/mpi)
- [MySQL](https://github.com/uxlfoundation/oneDAL/tree/main/samples/daal/cpp/mysql)

</details>

## Documentation

oneDAL documentation:

- [Release Notes](https://github.com/uxlfoundation/oneDAL/releases)
- [Get Started Guide](https://uxlfoundation.github.io/oneDAL/quick-start.html)
- [Developer Guide and Reference](https://uxlfoundation.github.io/oneDAL/)

Other related documentation:

- [daal4py documentation](https://intelpython.github.io/daal4py/)
- [Intel(R) Extension for Scikit-learn* documentation](https://uxlfoundation.github.io/scikit-learn-intelex/)
- [oneDAL Specifications](https://spec.oneapi.com/versions/latest/elements/oneDAL/source/index.html)

## Apache Spark MLlib

oneDAL library is used for Spark MLlib acceleration as part of [OAP MLlib](https://github.com/oap-project/oap-mllib) project and allows you to get a **3-18x** increase in performance compared to the default Apache Spark MLlib.

<img style="display:inline;" height=300 width=550 src="docs/readme-charts/intel%20oneDAL%20Spark%20samples%20vs%20Apache%20Spark%20MLlib.png"></a>

>*Technical details: FPType: double; HW: 7 x m5.2xlarge AWS instances; SW: Intel DAAL 2020 Gold, Apache Spark 2.4.4, emr-5.27.0; Spark config num executors 12, executor cores 8, executor memory 19GB, task cpus 8*

## Scaling

oneDAL supports distributed computation mode that shows excellent results for strong and weak scaling:

oneDAL K-Means fit, strong scaling result | oneDAL K-Means fit, weak scaling results
:-------------------------:|:-------------------------:
![](docs/readme-charts/Intel%20oneDAL%20KMeans%20strong%20scaling.png)  |   ![](docs/readme-charts/intel%20oneDAL%20KMeans%20weak%20scaling.png)

>*Technical details: FPType: float32; HW: Intel Xeon Processor E5-2698 v3 @2.3GHz, 2 sockets, 16 cores per socket; SW: Intel® DAAL (2019.3), MPI4Py (3.0.0), Intel® Distribution Of Python (IDP) 3.6.8; Details available in the article https://arxiv.org/abs/1909.11822*

## Governance

The oneDAL project is governed by the UXL Foundation and you can get involved in this project in multiple ways. It is possible to join the [AI Special Interest Group (SIG)](https://github.com/uxlfoundation/foundation/tree/main/ai) meetings where the group discuss and demonstrates work using this project. Members can also join the Open Source and Specification Working Group meetings.

You can also join the mailing lists for the [UXL Foundation](https://lists.uxlfoundation.org/g/main/subgroups) to be informed of when meetings are happening and receive the latest information and discussions.

You can contribute to this project and also contribute to the specification for this project, read the [CONTRIBUTING](CONTRIBUTING.md) page for more information.


## Support

Ask questions and engage in discussions with oneDAL developers, contributers, and other users through the following channels:

- [GitHub Discussions](https://github.com/uxlfoundation/oneDAL/discussions)
- [Community Forum](https://community.intel.com/t5/Intel-oneAPI-Data-Analytics/bd-p/oneapi-data-analytics-library)

You may reach out to project maintainers privately at onedal.maintainers@intel.com.

### Security <!-- omit in toc -->

To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

### Contribute <!-- omit in toc -->

We welcome community contributions. Check our [contributing guidelines](CONTRIBUTING.md) to learn more. You can also contact the oneDAL team via [UXL Foundation Slack] using
[#onedal] channel.

[UXL Foundation Slack]: https://slack-invite.uxlfoundation.org/
[#onedal]: https://uxlfoundation.slack.com/channels/onedal

## License <!-- omit in toc -->

oneDAL is distributed under the Apache License 2.0 license. See [LICENSE](LICENSE) for more information.

[oneMKL FPK microlibs](https://github.com/uxlfoundation/oneDAL/releases/tag/Dependencies)
are distributed under Intel Simplified Software License.
Refer to [third-party-programs-mkl.txt](third-party-programs-mkl.txt) for details.
