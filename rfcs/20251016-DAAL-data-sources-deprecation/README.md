# Deprecation of certain DAAL data sources in oneDAL 2025.1X

## Introduction

DAAL provides comprehensive data import capabilities from multiple sources,
including text files, in-memory data (RAM), and various database systems.

These [data sources](https://uxlfoundation.github.io/oneDAL/daal/data-management/data-sources.html)
wrap or convert data from source-specific formats
into DAAL's numeric table interface, making it accessible to computational
algorithms.

However, the data ingestion landscape has evolved with the development
of scikit-learn-intelex, which has established Python as the primary interface
for DAAL algorithms.

Python's ecosystem of data manipulation libraries, including pandas, parquet,
and specialized database connectors, now provides improved flexibility
and functionality for data import tasks.
This evolution has reduced the value of DAAL's native data source
implementations.

This shift enables users to leverage Python's extensive data handling
capabilities while continuing to benefit from DAAL's optimized computational
algorithms, creating a more streamlined workflow.

## Proposal

The following data sources will be marked as deprecated in oneDAL 2025.1X
with complete removal planned for the next major release, oneDAL 2026.0:

- [ODBC Data Source](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/data_management/data_source/odbc_data_source.h#L97)
- [KDB Data Source](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/data_management/data_source/kdb_data_source.h#L92)
- [MySQL Data Source](https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/include/data_management/data_source/mysql_feature_manager.h#L50)
