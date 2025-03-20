# oneAPI Data Analytics Library MySQL\* Samples

MySQL\* samples for the oneAPI Data Analytics Library (oneDAL) are designed to show how to use this library with a MySQL database in a C++ application.

Unzip the archive with oneDAL samples to your working directory (`<sample_dir>`).

## System Requirements
You can use oneDAL MySQL\* samples on Linux\*, Windows\*, and macOS\* operating systems. For a list of oneDAL hardware and software requirements, refer to release notes for the version of oneDAL you are using.

### MySQL\* implementations against which oneDAL has been validated:
- MySQL 5.6.22

**Note:** oneDAL is expected to work on other MySQL version as well. Let us know if you have any troubles with the distribution you are using.

## Preparation Before Build and Run
### MySQL support
You can download and install the MySQL\* application from [the official web page][mysql]. To be able to use MySQL\* C++  samples, make sure to configure the ODBC connector for the user who has permissions to create tables in the database. Also, make sure to replace the `mySQL_test` and `mySQL_test_32` database names in the `datasource_mysql.cpp` sample file with the actual database name you plan to use:

If your ODBC connector on Windows\* is installed in a directory different from `C:\Program Files (x86)\Windows Kits\8.1\Lib\winv6.3\um\`, make sure to update the `ODBC_PATH` variable in the `launcher.bat` script with the correct path before running the script.

### Setting Up the Build Environment
Before you build the sample, you must set certain environment variables that define the location of related libraries. The oneDAL includes the `vars` scripts that you can run to set environment variables:

- On Windows\*, you can find the `vars.bat` batch file at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\windows\daal\bin\:
vars.bat`
- On Linux\*, you can find the `vars.sh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\linux\daal\bin:
source vars.sh`
- On macOS\*, you can find the `vars.sh` shell script at `<install-dir>\compilers_and_libraries_xxxx.x.xxx\mac\daal\bin:
source vars.sh`

For more information about setting environment variables and configuring oneDAL, refer to Getting Started guides for the library.

## Build and Run Instructions
### On Windows\*
To build oneDAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `launcher` command with the `build` parameter:

```
cd <sample_dir>\cpp\mysql

launcher.bat build
```

The command creates the `.\_results\intel64` directory and builds `*.exe` executables and `*.exe` libraries, as well as creates a log file for build results.

To run oneDAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `launcher` command with the `run` parameter:

```
cd <sample_dir>\cpp\mysql

launcher.bat run
```

Select the same architecture parameter as you provided to the `launcher` command with the `build` parameter.

For each sample, the results are placed into the `.\_results\intel64\<sample name>\.res` file, depending on the specified architecture.

### On Linux\*
To build oneDAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/mysql

make {libintel64|sointel64} compiler={intel|gnu} mode=build
```

From the `{libintel64|sointel64}` parameters, select the one that matches the architecture parameter you provided to the `vars.sh` script and that has the prefix that matches the type of executables you want to build: `lib` for static and `so` for dynamic executables.

The command creates a directory for the chosen compiler, architecture, and library extension (`a` or `so`). For example: `_results/intel_intel64_a`.

To run oneDAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `make` command in the run mode. For example, if you run the `vars` script with the `intel64` target:

```
cd <sample_dir>/cpp/mysql

make libintel64 mode=run
```

The `make` command builds a static library for the Intel(R) 64 architecture and runs the executable.

### On macOS\*
To build oneDAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `make` command:

```
cd <sample_dir>/cpp/mysql

make {libintel64|dylibintel64} compiler={intel|gnu|clang} mode=build
```

From the `{libintel64|dylibintel64}` parameters, select the one that matches the architecture parameter you provided to the `vars.sh` script and that has the prefix that matches the type of executables you want to build: `lib` for static and `dylib` for dynamic executables.

The command creates a directory for the chosen compiler, architecture, and library extension (`a` or `dylib`). For example: `_results/intel_intel64_a`.

To run oneDAL MySQL C++ samples, go to the C++ MySQL samples directory and execute the `make` command in the run mode. For example, if you run the `vars` script with the `intel64` target:

```
cd <sample_dir>/cpp/mysql

make libintel64 mode=run
```

The `make` command builds a static library for the Intel(R) 64 architecture and runs the executable.

## Notices and Disclaimers

Performance varies by use, configuration and other factors. Learn more at www.Intel.com/PerformanceIndex​.  

No product or component can be absolutely secure. 

Your costs and results may vary.

Intel technologies may require enabled hardware, software or service activation.

**&copy; Intel Corporation**. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  Other names and brands may be claimed as the property of others.

\*Other names and brands may be claimed as the property of others.

<!-- Links -->
[mysql]: http://dev.mysql.com/downloads/
