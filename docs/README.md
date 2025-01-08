<!-- file: README.md
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

# Authoring and Editing

## Cross references (internal links)

To create a cross reference from one page of this document to another page,
ensure that an explicit target is used so that the cross reference is pointing
to a unique identifier, rather than a heading that could be repeated somewhere
else in the document.

The syntax for explicit targets is as follows:

For the target:

```none
.. _my-page:

My Page of Information
~~~~~~~~~~~~~~~~~~~~~~

```


For the cross reference:

```none
To learn more about my information, see :ref:`My Page of Information <my-page>`
```

To learn more about explicit targets, see:
<https://docs.readthedocs.io/en/stable/guides/cross-referencing-with-sphinx.html#explicit-targets>


# Build oneDAL documentation

Our documentation is written in restructured text markup and built with [Sphinx](http://www.sphinx-doc.org/en/master/).

## Software requirements

- Python 3.7.0 (may or may not work with older Python*, untested)
- `pip` package manager

## Generate documentation

To build oneDAL documentation locally:

1. Clone the repository:

		git clone https://github.com/uxlfoundation/oneDAL.git

2. Go to `docs` folder:

		cd oneDAL/docs

3. Install requried Python packages using `pip`:

		pip install -r requirements.txt

4. Run in the command line:

		make html

	You can find documentation in `build/html` folder.

	**Note:** By default, the documentation is generated with the Intel-branded library name.
	To generate the output for GitHub pages, run the following instead:

		make html-github