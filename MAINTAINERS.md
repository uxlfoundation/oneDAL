<!--
******************************************************************************
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
*******************************************************************************/-->

# Introduction

This document defines roles in oneDAL project.

# Roles and responsibilities

oneDAL project defines three main roles:
 * [Contributor](#contributor)
 * [Code Owner](#code-Owner)
 * [Maintainer](#maintainer)

These roles are merit based. Refer to the corresponding section for specific
requirements and the nomination process.

## Contributor

A Contributor invests time and resources to improve oneDAL project.
Anyone can become a Contributor by bringing value in one of the following ways:
  * Answer questions from community members.
  * Submit feedback to design proposals.
  * Review and/or test pull requests.
  * Test releases and report bugs.
  * Contribute code, including bug fixes, features implementations,
and performance optimizations.
  * Contribute design proposals.

Responsibilities:
  * Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
  * Follow the project [contributing guidelines](CONTRIBUTING.md).

Privileges:
  * Eligible to become Code owner/maintainer.

## Code Owner

A Code Owner has responsibility for a specific project component or a functional
area. Code Owners are collectively responsible, with other Code Owners, 
for developing and maintaining their component or functional areas, including
reviewing all changes to their their areas of responsibility and indicating
whether those changes are ready to merge. They have a track record of
contribution and review in the project.

Responsibilities:
  * Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
  * Follow and enforce the project [contributing guidelines](CONTRIBUTING.md).
  * Co-own with other code owners a specific component or aspect of the library,
    including contributing bug fixes, implementing features, and performance
    optimizations.
  * Review pull requests in their specific areas of responsibility.
  * Monitor testing results and flag issues in their specific areas of
    responsibility.
  * Support and guide Contributors.

Requirements:
  * Experience as Contributor for at least 6 months.
  * Track record of accepted code contributions to a specific project component.
  * Track record of contributions to the code review process.
  * Demonstrated in-depth knowledge of the architecture of a specific project
    component.
  * Commits to being responsible for that specific area.

Privileges:
  * PR approval counts towards approval requirements for a specific component.
  * Can promote fully approved Pull Requests to the `main` branch.
  * Can recommend Contributors to become Code Owners.
  * Eligible to become a Maintainer.

The process of becoming a Code Owner is:
1. A Contributor is nominated by opening a PR modifying the MAINTAINERS.md file
including name, Github username, and affiliation.
2. At least two specific component Maintainers approve the PR.


## Maintainer
Maintainers are the most established contributors who are responsible for the 
project technical direction and participate in making decisions about the
strategy and priorities of the project.

Responsibilities:
  * Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
  * Follow and enforce the project [contributing guidelines](CONTRIBUTING.md)
  * Co-own with other component Maintainers on the technical direction of a specific component.
  * Co-own with other Maintainers on the project as a whole, including determining strategy and policy for the project.
  * Suppport and guide Contributors and Code Owners.

Requirements:
  * Experience as a Code Owner for at least 12 months.
  * Track record of major project contributions to a specific project component.
  * Demonstrated deep knowledge of a specific project component.
  * Demonstrated broad knowledge of the project across multiple areas.
  * Commits to using priviledges responsibly for the good of the project.
  * Is able to exercise judgment for the good of the project, independent of
    their employer, friends, or team.

Privileges:
  * Can represent the project in public as a Maintainer.
  * Can promote Pull Requests to release branches and override mandatory
  checks when necessary.
  * Can recommend Code Owners to become Maintainers.

Process of becoming a maintainer:
1. A Maintainer may nominate a current code owner to become a new Maintainer by 
opening a PR against MAINTAINERS.md file.
2. A majority of the current Maintainers must then approve the PR.

# Code Owners and Maintainers List

## oneDAL core

Team: @uxlfoundation/onedal-write

Currently entire Intel oneDAL team serves at contributor level and have corresponding rights
with additional separation of roles coming with repository migration to UXL


### oneDAL Architecture
| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Victoriya Fedotova | @Vika-F               | Intel Corporation | Maintainer |
| Aleksandr Solovev  | @Alexandr-Solovev     | Intel Corporation | Maintainer |
| Alexander Andreev  | @Alexsandruss         | Intel Corporation | Maintainer | 

### AArch64

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Rakshith G B       | @rakshithgb-fujitsu   | FUjitsu           | Code Owner |

### RISC-V

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Keeran Rothenfusser| @keeranroth           | Rivosinc          | Code Owner |


### Release management

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Nikolay Petrov     | @napetrov             | Intel Corporation | Maintainer |
| Sergey Yakovlev    | @syakov-intel         | Intel Corporation | Maintainer |
| Maria Petrova      | @maria-Petrova        | Intel Corporation | Maintainer |


