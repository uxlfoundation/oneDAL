.. Copyright 2014 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

.. highlight:: cpp

Coding Guidelines
^^^^^^^^^^^^^^^^^

As more time and effort is spent on maintenance rather than on writing the initial code,
it is essential to keep the code base manageable.
Any developer should be able to quickly understand the code written by another developer.
By ensuring **consistency** in the code base you save time and effort for both yourself and others.

These guidelines cover our coding style along with more practical issues.
To learn more about contribution process, see `How to Contribute <https://github.com/uxlfoundation/oneDAL/blob/main/CONTRIBUTING.md>`_.

Coding Style
============

- `General rules`_
- `Naming conventions`_
- `Declaration order`_
- `Typedefs`_
- `Defines`_
- `Common definitions`_
- `Statements`_
- `Comments`_
- `Templates declaration`_
- `Deprecated API`_

General Rules
*************

Whitespaces
-----------

- Do not leave trailing spaces in the end of a line.
- Always put an empty line in the end of the file.
- Surround operators with spaces: ``x += 1`` is preferred to ``x+=1``.
- Only use spaces, not tabs.
- Keep the same indent in spaces: indent is equal to four spaces.

.. tip::

  Configure your editor to produce spaces instead of tabs.
  This guarantees that source code always looks and prints fine regardless of any editor settings.

Functions
---------

- The order of function's parameters: first inputs, then outputs.

C++ Files
---------

Each C++ file should begin with the following:

- a comment containing the name of the file
- the copyright notice
- a brief description of the file's content

.. Attention::

  Code must compile without errors and warnings.

Header Files
------------

A header file should include only the header files it needs.

- Whenever possible put ``include``\s in .cpp files, not in header files.
- Do not use ``#include`` if a forward declaration is sufficient.

Naming Conventions
******************

Rules that apply for a name of any entity:

- Give descriptive and meaningful names.
- Avoid ambiguity.
- Do not abbriviate words.
- Do not remove letters from words.

.. container:: comparison

    .. container:: column

      .. admonition:: Not recommended
        :class: error

        ::

          int n; // Meaningless
          int nerr; // Ambiguous abbreviation
          int nCompConns; // Ambiguous abbreviation
          int wgcConnections; // Meaningless for others
          int pcReader; // Ambiguous ("pc" can mean different things)
          int cstmrId; // Missing letters

    .. container:: column

      .. admonition:: Good
        :class: hint

        ::

          int priceCountReader;
          int numErrors; // "num" is a widespread convention

For more specific rules, see:

- `File names`_
- `Type names`_
- `Function names`_
- `Variable names`_
- `Class member names and method names`_
- `Constant names`_
- Naming conventions for `distributed and streaming computational schemes`_

File Names
----------

The name of a file:

- should be in all lowercase
- may include underscores (``_``)

::

  algorithm/kmeans_assign.h

Type Names
----------

The names of classes, structs, typedefs, and enums:

- should start with a capital letter
- should have a capital letter for each new word
- should not contain underscores (``_``)

::

  class MyClass;
  struct MyStruct;
  typedef int MyType;
  enum MyEnum;

Function Names
--------------

The name of a function:

- should start with a lowercase letter
- should have a capital letter for each new word
- should not contain underscores (``_``)

::

  addTableEntry();
  deleteTable();

Variable Names
--------------

The name of a local variable:

- should start with a lowercase letter
- should have a capital letter for each new word
- should not contain underscores (``_``)

::

  int numBlocks;

Class Member Names and Method Names
-----------------------------------

There is a difference in naming ``public`` and ``private`` class members and methods:

- ``public`` class members and methods are named in the same way as local variables and functions:

  - `Variable Names`_
  - `Function Names`_

- ``private`` class members and methods should have an additional underscore (``_``) at the beginning.

::

  class Table
  {
  public:
    int colsInTable; // external
  private:
    string _tableName; // internal
  }

Constant Names
--------------

Constants:

::

  const int daysInAWeek = 7;

Enum members:

::

  enum MyTableTypes {intType, doubleType};

Distributed and streaming computational schemes
-----------------------------------------------

Enum Names
++++++++++

For numeric tables: ``InputId``. For models: ``Input[Name]Id``

- remove ``Distributed`` from names
- add ``Step?``
- remove ``Numeric Table`` and ``Table`` from names

Name template is the following: ``[Prefix][Method][SbjName][Type]Id;``

Prefix
  - ``Step?Master`` or ``Step?Local`` – for distributed processing mode
  - ``Online`` – for online processing mode
  - none – for batch processing mode

Method
  - SVD, Correlation, etc.

SbjName
  - PartialResult, Input

Type
  - ``Collection`` or ``Model``
  - none – for Numeric Table

Examples of renaming:

- ``MasterInputId`` -> ``Step2MasterInputId``
- ``DistributedPartialResultStep3Id`` -> ``Step3LocalPartialResultId``
- ``PartialSVDTableResultId`` -> ``Step?LocalSVDPartialResultId``
- ``Step3LocalNumericTableInputId`` -> ``Step3LocalInputId``
- ``Step3LocalCollectionInputId`` -> ``Step3LocalInputCollectionId``

Class Names
+++++++++++

Name templates:

- ``DistributedInput<step, method>``
- ``DistributedPartialResult<step, method>`` - for distributed computation
- ``PartialResult<method>`` - for online computation and distributed step1

step
  ``step2Master``, ``step3Local``, etc. (starts from 2)

method
  default if there is only one method used

PartialResult
  is returned from 1st computation step

DistributedPartialResult
  is returned from steps 2 and higher


Declaration Order
*****************

Use the specified order of declarations within a class:
your class definition should start with its ``public:`` section, followed by
its ``protected:`` section, and then its ``private:`` section. If any of these
sections are empty, omit them.

The declarations generally should be in the following order:

-  Typedefs and Enums
-  Constants (``static const`` data members)
-  Constructors
-  Destructor
-  Methods, including static methods
-  Data Members (except ``static const`` data members)

.. Important:: Friend declarations should always be in the ``private`` section.

Typedefs
********

Use typedefs for template-based classes. It improves code readability and ensures
that the same type is used in all cases.

::

  // typedefs
  typedef hash_map<UrlTableProperties *, string> PropertiesMap;

Defines
*******

::

  #define DAL_MY_DEFINE // external
  #define __MY_DEFINE // internal
  #ifndef __FILE_NAME_H__
  #define __FILE_NAME_H__
  #pragma directive should be indented

Common definitions
******************

::

  defined(__x86_64__) // defined for Linux 64bit and MacOS 64bit OSes
  defined(__linux__) // defined for Linux and MacOS
  defined(__APPLE__) // defined for MacOS
  defined(_WIN64) // defined for Winddows 64bit
  defined(_WIN32) // defined for Winddows 32bit & Winddows 64bit
  defined(__ICL) // defined for Intel compiler. Has numeric value.
  defined(__INTEL_COMPILER) // defined for Intel compiler. Has numeric value.
  defined(__INTEL_LLVM_COMPILER) // defined for Intel LLVM compiler
  defined(DAAL_INTEL_CPP_COMPILER) // defined for all Intel C++ compilers
  defined(_MSC_VER) // defined for Intel and MS compilers. Has numeric value.

Statements
**********

- Each line should contain at most one statement:

  .. container:: comparison

      .. container:: column

        .. admonition:: Not recommended
          :class: error

          ::

            // Not recommended
            a++; b++;

      .. container:: column

        .. admonition:: Good
          :class: hint

          ::

            a++;
            b++;

- The statement that follows conditional statement should be on a separate line:

  .. container:: comparison

      .. container:: column

        .. admonition:: Not recommended
          :class: error

          ::

            // Not recommended
            if (condition) statement;

      .. container:: column

        .. admonition:: Good
          :class: hint

          ::

            if (condition)
                statement;


Comments
********

Comments should provide information that is not readily understandable from the code itself.

Linus Torvalds `says <https://github.com/torvalds/linux/blob/master/Documentation/process/coding-style.rst#8-commenting>`__:

.. container::

  .. container:: quotation

      *NEVER try to explain HOW your code works in a comment: it's much
      better to write the code so that the \_working\_ is obvious, and it's
      a waste of time to explain badly written code.*

      *Generally, you want your comments to tell WHAT your code does, not
      HOW. Also, try to avoid putting comments inside a function body: if
      the function is so complex that you need to separately comment parts
      of it, you should probably split it into smaller functions. You can
      make small comments to note or warn about something particularly
      clever (or ugly), but try to avoid excess. Instead, put the comments
      at the head of the function, telling people WHAT it does, and
      possibly WHY it does it.*

Function Comments
-----------------

- Always document functions that are a part of an API.
- Document locally used functions when they require clarification.

Variable Comments
-----------------

In most cases, the name of a variable should be descriptive enough
to give a good idea of what the variable is used for. In certain cases,
more comments are required.

Class Data Members Comments
---------------------------

- Each data member of a class should have a comment describing what it is used for.
- If the variable can take sentinel values with special meanings, such as a null pointer or -1, document this.

::

  private:
  // Keeps track of the total number of entries in the table.
  // Used to ensure we do not go over the limit. -1 means
  // that we don't yet know how many entries the table has.
  int _nEntries;

Function Declarations Comments
------------------------------

- Every declaration of a non-trivial function should be preceded by comments that describe what the function does and how to use it.

Templates declaration
*********************

- Empty template classes should have a ``{}``.
- ``Template<...>`` should all be in one line.

Deprecated API
**************

To mark API in |short_name| as deprecated, do the following:

.. tabs::

  .. tab:: C++

    1. Mark deprecated API with ``DAAL_DEPRECATED`` define. For virtual functions use ``DAAL_DEPRECATED_VIRTUAL`` define.

      ::

        DAAL_DEPRECATED class A // Deprecated class
        {
        public:
          DAAL_DEPRECATED A() {} // Deprecated class constructor
          DAAL_DEPRECATED void func() {} // Deprecated function in class
          DAAL_DEPRECATED int classMember; // Deprecated class member
        };


    2. Add a special tag to documentation comment:

      - Use ``\DAAL_DEPRECATED`` when you remove something from API.
      - Use ``\DAAL_DEPRECATED_USE{ newFunction }`` when you introduce a new function to use instead of a deprecated one and want to reference it.

      These tags add standard phrases about deprecation. They are defined in our doxygen configurations.

      .. code-block::
        :emphasize-lines: 3,9

        /**
        * Description for function that will be removed
        * \DAAL_DEPRECATED
        */
        DAAL_DEPRECATED int removedFunction() {}

        /**
        * Description for function with old name
        * \DAAL_DEPRECATED_USE{ newFunction }
        */
        DAAL_DEPRECATED int oldFunction() {}

        /**
        * Description for new function
        */
        void newFunction() {}

    .. note::

      To mark enums and their elements as deprecated, you only need to put the appropriate documentation tag in a doc comment.

Notes
-----

You can generate a reference to another functionality without ``\ref``:

1. Add the appropriate tag to documentation comment :

  - Use ``\DAAL_DEPRECATED_USE{ newFunction }`` to reference global functions.
  - Use ``\DAAL_DEPRECATED_USE{ ClassName::newFunction }`` to reference functions that are defined inside classes.
  - Use ``\DAAL_DEPRECATED_USE{ \ref daal::<all namespaces to ClassName>::ClassName::newFunction "newFunction" }`` if a reference
    is not parsed correctly by doxygen (quotation marks are a part of a command).

2. Add ``DAAL_DEPRECATED`` and ``DAAL_DEPRECATED_VIRTUAL`` after ``\return`` tag.

Programming guidelines
======================

Local Variables
***************

To make it easier for the reader to find the declaration, see what type the variable is, and what it was initialized to:

- Place a function's variables in the narrowest scope possible.
- Initialize variables in the declaration.
- Use initialization instead of declaration and assignment.

  .. container:: comparison

      .. container:: column

        .. admonition:: Not recommended
          :class: error

          ::

            int i;
            i = f();

      .. container:: column

        .. admonition:: Good
          :class: hint

          ::

            // Declaration includes initialization.
            int j = g();


  .. container:: comparison

      .. container:: column

        .. admonition:: Not recommended
          :class: error

          ::

            vector<int> v;
            v.push_back(1); // Prefer initializing using brace initialization.
            v.push_back(2);

      .. container:: column

        .. admonition:: Good
          :class: hint

          ::

            // Good -- v starts initialized, brace initialization is used.
            vector<int> v = {1, 2};


When you work with variables needed for ``if``, ``while``, and ``for`` statements:

- If the variable is not an object, declare it within the statement to confine it to the scope of the statement:

  ::

    while (const char* p = strchr(str, '/'))
        str = p + 1;

- If the variable is an object, declare it outside the loop.

  As object's constructor is invoked every time the object is created or enters the scope, and its destructor is
  invoked every time the object goes out of scope, you make your code inefficient by placing variable's declaration inside a loop.

  .. container:: comparison

      .. container:: column

        .. admonition:: Not recommended
          :class: error

          ::

            // Inefficient implementation:
            for (int i = 0; i < 1000000; ++i)
            {
                Foo f; // My ctor and dtor get called 1000000 times each.
                f.DoSomething(i);
            }

      .. container:: column

        .. admonition:: Good
          :class: hint

          ::

            // More efficient
            Foo f; // My ctor and dtor get called once each.
            for (int i = 0; i < 1000000; ++i)
            {
                f.DoSomething(i);
            }

Constants
*********

- Use constant variables or functions returning the constant value instead of hard coded constant values:

  .. container:: comparison

      .. container:: column

        .. admonition:: Not recommended
          :class: error

          ::

            foo("Hard coded string");
            // ...
            bar("Hard coded string");

      .. container:: column

        .. admonition:: Good
          :class: hint

          ::

            const char* hardCodedString() { return "Hard coded string"; }
            foo(hardCodedString());
            bar(hardCodedString());

Static and Global Variables
***************************

.. Caution::

  Static variables of class type may cause hard-to-find bugs
  due to the `undetermined order of construction and destruction`_.

- Avoid using static variables of class type.
- Use `static variables within a function scope`_ instead.

Undetermined order of construction and destruction
--------------------------------------------------

The order in which class constructors and initializers for
static variables are called is only partially specified in C++. It may
even change from build to build, which causes bugs that are difficult to find.

On program termination, global and static variables are destroyed.
The order in which destructors are called is defined to be
the reverse of the order in which the constructors were called. Since the order in which
constructors are called is undertermined, so is the order of destructors.

Consider the following examples:

1. At program-end time, a static variable is destroyed, but the code is still running in another thread,
   tries to access the destroyed variable and, therefore, fails.

2. The destructor for a static string variable might be run prior to the destructor for another variable
   that contains a reference to that string.

Static variables within a function scope
----------------------------------------

A static variable within a function scope may be initialized with
the result of a function, since its initialization order is well-defined
and does not occur until control passes through its declaration.

This is why we reccomend using a static variable within a function scope
if you need a static or a global variable of a class type. Create a function
that returns the variable you need, but make sure that the function call
does not happen when the program is terminating.

.. container:: comparison

    .. container:: column

      .. admonition:: Not recommended
        :class: error

        ::

          static SingletonClass inst; // Bad – initialization order is undefined.


    .. container:: column

      .. admonition:: Good
        :class: hint

        ::

          SingletonClass& getInst()
          {
              static SingletonClass inst;
              return inst;
          }
          // Good – initialization happens when the function is called

Usage of ``const``
******************

The general rule is to use ``const`` declaration whenever possible.

For details, see:

- `Usage of const in Functions`_
- `Usage of const Methods`_
- `Usage of const and mutable Data Members`_

The advantages of using ``const``:

- Implies semantic constraints handled by compiler
- Adds a level of compile-time type checking
- Helps to specify the logic of the code
- Helps to make the code consistent and self-documented
- Helps to find possible errors

::

  // Good
  void foo(Collection& c)
  {
    //we need to save initial size, then make sure it will not be modified
    const size_t len = c.size();
  }

Usage of ``const`` in Functions
-------------------------------

If a function guarantees that it will not modify an argument passed by
reference or by pointer, the corresponding function parameter should be
a reference-to-const (``const T&``) or pointer-to-const (``const T*``),
respectively.

::

  void func(const Foo& foo)
  {
    // foo should not be modified by this function, 'const' states it clearly
  }

Usage of ``const`` Methods
--------------------------

Declare methods to be ``const`` whenever possible:

- Accessors should almost always be ``const``.
- Other methods should be ``const`` if they do not modify any data members,
  do not call any non-const methods, and do not return a
  non-const pointer or non-const reference to a data member.

::

  class Foo
  {
  public:
    int getValue() const; // does not modify data members
    void setValue(int); // can modify data members
  protected:
  //...
  }

Usage of ``const`` and ``mutable`` Data Members
-----------------------------------------------

- Consider making data members ``const`` whenever they do not need to be modified after construction.

  ::

    class Foo
    {
    public:
      Foo (int val): _val(val){} // never going to be changed
    protected:
      const int _val;
    }

``const`` usage is sort of is viral: once appeared in a function it
causes its propagation in related functions. But this is an excellent
feature!

- If a class needs to modify its member in a ``const`` function (e.g. when
  implementing caching), then declare this data member ``mutable``.

  ::

    class Foo
    {
    public:
      Foo():_isInitialized(false){}
      int get () const
      {
          if(!_isInitialized)
          {
              _value = calcValue();
              _isInitialized = true;
          }
          return _value;
      }
    private:
      mutable int _value;
      mutable bool _isInitialized;
    }

  .. Caution::

    When you use ``mutable`` in multi-threaded program, make
    sure thread-safe access is provided.

Ownership (RAII – Resource Acquisition Is Initialization)
*********************************************************

C++ provides constructor and destructor symmetry which can naturally be
used to manage resource allocation and deallocation pairs, e.g.
dynamically allocated memory. This bookkeeping technique is called
**Resource Acquisition Is Initialization (RAII)**.

Holding a resource is tied to the object's lifetime:

- Resource allocation (acquisition) is done by the constructor during object's creation.
  More specifically, during initialization.

- Resource deallocation (release) is done by the destructor during object's destruction.
  If objects are destroyed properly, resource leaks do not occur.

  ::

    class FileHolder
    {
    public:
      FileHolder (FILE* f): _file(f)
      {
      }
      ~FileHolder
      {
        if(_file)
          fclose(_file);
      }
    private:
      FILE* _file;
    }

- Place resource acquisition in a separate instruction:

  .. container:: comparison

      .. container:: column

        .. admonition:: Not recommended
          :class: error

          ::

            // Not recommended
            void foo(Foo* data, int priority);
            int priority();

            foo(std::shared_ptr<Foo>(new Foo()), priority());
            // if priority() throws exception then the memory allocated by new can be lost

      .. container:: column

        .. admonition:: Good
          :class: hint

          ::

            // Preferable
            void foo(Foo* data, int priority);
            int priority();

            std::shared_ptr<Foo> ptr(new Foo());
            foo(ptr, priority());


Smart Pointers
**************

"Smart" pointers are partial case of RAII idiom. These classes act like pointers, e.g. by overloading the ``*`` and ``->``
operators providing additional features, such as automatic memory management.
Some smart pointer types can be used to automate ownership bookkeeping.

Passing parameters of non-primitive types by reference
******************************************************

- Parameters of complex types should be passed by reference.
- Use `const` to emphasize they are not modified.

This makes code more efficient: copy constructors are not called. It also allows to avoid unwanted inexplicit casting of types.

.. admonition:: Not recommended
  :class: error

    ::

      // Not recommended
      class string
      {
      public:
        string(const string& other);// allocates memory buffer and copies ‘other’ content into it.
      };

      void print(string s) {// Inefficient: string(string) is called
        std::cout << s;
      }

.. admonition:: Not recommended
  :class: error

    ::

      // Not recommended
      class Base
      {
        Base(const Base& other);
        virtual printMe() const { std::cout << "Base"; }
      };

      class Derived: public Base
      {
        virtual printMe() const { std::cout << "Derived"; }
      };

      void print(Base b); // Reference was forgotten!!!

      Derived d;
      print(d); // Logical error: Base was created, polymorphic behavior is lost!


Preprocessor Macros
*******************

.. Caution::

  Be very cautious with macros. Prefer inline functions, enums,
  and const variables to macros.

Macros mean that the code you see is not the same as the code the
compiler sees. This can introduce unexpected behavior, especially since
macros have global scope.

- Use a ``const`` or ``enum`` instead of ``#define NAME value``: 

  ::

    const unsigned int MAX_CAPACITY = 1000;

- If constants define a related set, make them an enumerated type:

  ::

    class MyNet
      {
      public:
          enum { eMaxUsers = 255; }
          ...
      private:
          enum { eMaxServers = 10; }
      };


Namespaces
**********

- Use namespaces instead of name prefixes to prevent name conflicts.
- Never put ``using`` directive before ``#include`` or in header files
  outside of functions, methods, or classes.

  .. Caution::

    Since you do not know in what order directives may appear in the code below,
    this can leas to a name confict.

- You may use a ``using`` declaration anywhere in a .cpp file.

.. admonition:: Not recommended
  :class: error

  ::

    // Not recommended

    // Header foo.h
    namespace foo
    {
      class Bar{}
    }
    using namespace foo;

.. admonition:: Not recommended
  :class: error

  ::

    // client_code.cpp
    #include “foo.h”
    struct Bar { int code; }

    Bar res; // Error: ambiguous


Functions
*********

- `Write short functions`_.
- Avoid multiple nesting levels, e.g. use small functions for complex conditions checking.
- Never write code that depends on the order of `function's arguments`_.

Write short functions
---------------------

Prefer small and focused functions, as they promote clarity and correctness.
Long functions are sometimes appropriate, so no hard limit is placed on a
function's length.

.. note::

  If a function is longer then 40 lines, consider breaking it into smaller
  and easier-to-maintain pieces of code. Make sure that by doing this you are not
  harming the structure of the program.

Linus Torvalds `says <https://github.com/torvalds/linux/blob/master/Documentation/process/coding-style.rst#6-functions>`_:

.. container::

  .. container:: quotation

    *Functions should be short and sweet, do just one thing and do that
    well.*

    *The maximum length of a function is inversely proportional to the
    complexity and indentation level of that function. So, if you have a
    conceptually simple function that is just one long (but simple)
    case-statement, where you have to do lots of small things for a lot
    of different cases, it's OK to have a longer function.*

    *Another measure of the function is the number of local variables.
    They shouldn't exceed 5-10, or you're doing something wrong. Re-think
    the function, and split it into smaller pieces. A human brain can
    generally easily keep track of about 7 different things, anything
    more and it gets confused. You know you're brilliant, but maybe you'd
    like to understand what you did 2 weeks from now.*

Function's arguments
--------------------

Do not rely on the evaluation order of function’s arguments, as it is undefined.
Directly specify evaluation order instead.

.. container:: comparison

    .. container:: column

      .. admonition:: Not recommended
        :class: error

        ::

          // Not recommended

          void foo(int, int);
          int count = 5;
          foo(++count, ++count); // Error: arguments evaluation order is undefined

          int inc(int &val) { return ++val; }
          foo(inc(count), inc(count)); // Error: arguments evaluation order is undefined

    .. container:: column

      .. admonition:: Good
        :class: hint

        ::

          int count = 5;
          int tmp = ++count;
          foo(tmp, ++count);

Infix and postfix increment operators
*************************************

- Prefer infix increment operator to the postfix one.

When a variable is incremented (``++i`` or ``i++``) or decremented (``--i`` or ``i--``)
and the value of the expression is not used, one must decide whether to
pre-increment (pre-decrement) or post-increment (post-decrement).

When the return value is ignored, the "pre" form (``++i``) is never less
efficient than the "post" form (``i++``), and is often more efficient. This
is because post-increment (post-decrement) requires a copy of ``i`` to be
made, which is the value of the expression. If ``i`` is an iterator or other
non-scalar type, copying ``i`` could be expensive. Since the two types of
increment behave the same when the value is ignored, why not just always
pre-increment?

.. container:: comparison

    .. container:: column

      .. admonition:: Not recommended
        :class: error

        ::

          for(size_t i = 0; i < n; i++) // Not recommended

    .. container:: column

      .. admonition:: Good
        :class: hint

        ::

          for(size_t i = 0; i < n; ++i) // Preferable

Assertions
**********

Assert is a powerful tool. It makes code self-documented and self-checking in run-time. Some code quality standards define asserts
usage paradigm "the more the better", i.e. the number of asserts in the code characterizes its quality.

- Use ``assert`` to check invariants and internal assumptions.
- Make sure that ``assert`` usage does not cause side effects:

.. admonition:: Not recommended
      :class: error

      ::

        // Not recommended
        assert( ++i < limit); // Error: ++i happens in debug only

Classes
*******

-  Use classes with strictly defined purposes.
-  Avoid fat-interface classes.
-  Do not add a new method to the class if it is not supposed to modify its protected data nor does it call its protected methods.
-  Use friends only when it is absolutely necessary.
-  Declare virtual destructor in a polymorphic base class.

Initialization
--------------

If you define member variables inside a class, provide an in-class initializer for each member variable or write a constructor (it can be a default constructor).

.. caution::

  If you do not declare any constructors yourself, then the compiler will generate a default constructor for you.
  This might result in some fields not being initialized at all or initialized to inappropriate values.

Initializers
++++++++++++

Use initializers list instead of assignment in constructors:

.. container:: comparison

    .. container:: column

      .. admonition:: Not recommended
        :class: error

        ::

          // Not recommended

          class Foo
          {
            Foo(){
              _string = "AAA";
            }
            // This is an equivalent to
            // _Foo(): string() { _string = "AAA"; }
          };

    .. container:: column

      .. admonition:: Good
        :class: hint

        ::

          // The following is more efficient:
          class Foo
          {
            Foo():_string("AAA"){} // string(const char*) only
          };


**Exception:** Acquisition of resource allocated by ``new`` is preferable to perform as assignment in the constructor body:

.. admonition:: Not recommended
  :class: error

  ::

    // Not recommended
    class Foo
    {
      Foo(): _ptrA(new A()), _ptrB(new B()){
      }
    };
    // Memory can be allocated first, then passed to _ptrA and _ptrB.
    // If B::B() throws exception, then memory allocated by new A() is lost


Constructors
++++++++++++

Avoid doing complex initialization in constructors (in particular, initialization that can fail or that requires virtual method calls).

The problems with doing work in constructors are:

-  There is no easy way for constructors to signal errors, short of using exceptions.
-  If the work fails, you now have an object which initialization code have failed, so it may be an indeterminate state.
-  If the work calls virtual functions, these calls will not get dispatched to the subclass implementations. Future modifications of
   your class can quietly introduce this problem even if your class is not currently sub-classed.
-  If someone creates a global variable of this type (which is against the rules, but still), the constructor code will be called
   before ``main()``, possibly breaking some implicit assumptions in the constructor code.

Constructors should never:

- call virtual functions
- attempt to raise non-fatal failures.

.. note:: If your object requires non-trivial initialization, consider using a factory function or ``init()`` method.

.. note:: Destructors should not call virtual functions or throw exceptions either.

Constructors for non-trivial types
++++++++++++++++++++++++++++++++++

Use explicit copy constructors for non-trivial types:

::

  class Base
  {
    explicit Base(const Base& other);
    virtual printMe() const { std::cout << "Base"; }
  };

  class Derived: public Base
  {
    virtual printMe() const { std::cout << "Derived"; }
  };

  void print(Base b);

  Derived d;
  print(d); // Compile time error

This helps to prevent unwanted conversion to the base type.
It also helps to indicate the places where duplication of the class data is required, which can probably be handled more efficiently.

Clone all parts of an object
----------------------------

Call base class copy constructor and copy constructors for all members of the class.
The same goes for assignment operation.

Assignment
----------

Operator ``=`` should check for 'this' argument passed and return a reference to the object.

New and delete
--------------

- Provide overloaded ``new`` and ``delete`` consistently: overloaded ``void operator new(params)`` should be
  matched with ``void operator delete(void*, params)``, the same goes for ``new []``, ``delete []``.
- Provide all standard versions of ``new()`` operator.

Access Control
--------------

- Make data members private, and provide access to them through ``set`` and ``get`` methods as needed.
- Do not open data except for static const values on a class. Exception: simple structures.

Template metaprogramming
------------------------

- Avoid complicated template programming. It allows extremely flexible interfaces that are type safe and high performance,
  but, at the same time, the techniques used in template metaprogramming are often obscure and hard to maintain.
- Put considerable effort into minimizing and isolating the complexity.

  - Make sure that tricky code is especially well commented: carefully document how the code is used, say something about what the "generated" code looks like.
  - Pay extra attention to the error messages that the compiler emits when users make mistakes.
    The error messages are a part of your user interface, and your code should be tweaked as necessary so that the error messages are understandable and helpful.

- Use class member types (``typedefs``) to increase readability of template classes:

  ::

    template<typename T>
    class Point
    {
    public:
      typedef T value_type;
      typedef std::shared_ptr<Point> pointer;
      ...

      static pointer create() { return pointer(new Point()); }
    };


Hiding the implementation in ``Model`` classes
**********************************************

Separate the interface and the implementation in ``Model`` classes. This
allows to change the implementation specifics without changing the public interface.

The implementation of the ``Model`` class should consist of the following three parts:

- a class with a public interface
- an internal class with the implementation
- an internal class that maps the interface to the implementation

Below you can find the details of what each class should consist of.

1. Public interface.

   This class defines the public API of the model via purely virtual functions and should not provide any implementation.

  ::

    namespace some_namespace
    {
    class Model : public daal::algorithms::Model
    {
    public:
        virtual int foo() = 0;
    };
    }


2. Internal class with the implementation.

   This class contains everything that is needed for implementation:

   - functionality of the public API of the model
   - training algorithm
   - prediction algorithm

  ::

    namespace some_namespace
    {
    namespace internal
    {
    class ModelInternal
    {
    public:
        int fooImpl() {...}
        /* All the methods that are needed to implement training and prediction algorithms go here */

    };
    }
    }


3. Internal class that maps the interface to the implementation.

   This class can be used inside the respective algorithm’s kernel. To unify the implementation of the methods of this class,
   ``typedef ImplType`` should be defined within this class and point to the type with the implementation.

  ::

    namespace some_namespace
    {
    namespace internal
    {
    class ModelImpl : public some_namespace::Model,
                      public some_namespace::internal::ModelInternal
    SDL{
    public:
        typedef some_namespace::internal::ModelInternal ImplType;

        int foo() DAAL_C11_OVERRIDE { return ImplType::fooImpl(); }
    };
    }
    }


SDL Requirements
****************

SDL stands for Security Development Lifecycle and discusses the best known practices:

- `Handling primitive data types`_
- `Working with pointers`_
- `Working with buffers`_
- `Handling exceptions`_
- `Handling null byte injection`_

Handling Primitive Data Types
-----------------------------

SDL requires to explicitly catch type conversion errors:

- Detect and handle runtime errors that could occur upon conversion.
- Detect invalid type conversions by using exceptions or post-validation code where applicable.

For |short_name| it means that every method where type conversion occurs
should return ``Status`` explicitly or by reference.

Below we discuss conversion between integer types in different cases:

- `Code is performance-oriented`_ (i.e. recursive functions, loops with indefinite or a large number of iterations)
- `Code is not performance-oriented`_

.. note:: Conversion of floating point data types is not a security problem.

.. important:: It is required to add error code into `error_handling.cpp <https://github.com/uxlfoundation/oneDAL/blob/main/cpp/daal/src/services/error_handling.cpp>`_.

Code is performance-oriented
++++++++++++++++++++++++++++

In this case, it is a good practice to use ``DAAL_ASSERT``:

::

  for (size_t i = 0; i < nrows; i += di)
  {
      if (i + di > nrows)
      {
          di = nrows - i;
      }

      for (size_t j = 0; j < ncols; ++j)
      {
          NumericTableFeature & f = (*_ddict)[j];
          char * pc = (char *)_arrays[j].get();
          DAAL_ASSERT(pc)
          char * ptr = pc + (idx + i) * f.typeSize;

          internal::getVectorUpCast(f.indexType, internal::getConversionDataType<T>())(di, ptr, lbuf);

          for (size_t k = 0; k < di; ++k)
          {
              buffer[(i + k) * ncols + j] = lbuf[k];
          }
      }
   }

Code is not performance-oriented
++++++++++++++++++++++++++++++++

For code that is not performance-oriented, consider whether or not the conversion is safe.

- `Conversion is safe`_
- `Conversion is not safe`_

.. note::

  Type conversion is safe if the range of the destination type contains the range of the source type.
  For example, the following would be considered a safe conversion:

  - from ``byte`` to ``int``
  - from ``byte`` to ``size_t``
  - from ``short`` to ``int``
  - from ``unsigned short`` to ``unsigned int``

  See `examples of safe conversions`_ for more details.

Conversion is safe
~~~~~~~~~~~~~~~~~~

SDL allows two options:

- a comment justifying why the type converstion is safe for each instance where it is used
- ``DAAL_ASSERT``

.. note:: Using ``DAAL_ASSERT`` is a better solution as macros is more informative than a comment.

Conversion is not safe
~~~~~~~~~~~~~~~~~~~~~~

In this case, every method where type conversion occurs should return ``Status``.

Examples of type conversion verification
++++++++++++++++++++++++++++++++++++++++

In the examples below, ``<error code>`` depends on the context.

Conversion from ``int`` to ``size_t``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  int blockIndex;
  // ...
  DAAL_CHECK(blockIndex >= 0, <error code>)
  size_t N = (size_t)blockIndex;

Conversion from ``size_t`` to ``int``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

  size_t blockIndex;
  // ...
  DAAL_CHECK(blockIndex <= INT_MAX, <error code>)
  int N = (int)blockIndex;

Examples of safe conversions
++++++++++++++++++++++++++++

The conversion is safe if a variable being converted is declared as ``const`` and the following is true for the value it is initialized with:

- It is unrelated to the input data.
- It is within the range of the destination type.

  ::

    const int n = 1024
    DAAL_ASSERT(n >= 0)
    size_t uN = (size_t)n;

The conversion is safe if the following is true for the value that a variable being converted is initialized with:

- It is with the ragne of the destination type
- It cannot leave that range during the workflow.

  ::

    size_t getMaxElement(const int *elementsArray, size_t nElements) const
    {
            status = services::Status();
            int max = 0;
            for(size_t i = 0; i < nElements; i++)
            {
                if (max < elementsArray[i])
                {
                    max = elementsArray[i];
                }
            }
            // max is initialize by 0 and during workflow it can increase only,
            // so max is equal to 0 or more than 0,  such conversion is safe
            DAAL_ASSERT(max >= 0)
            return (size_t)max;
    }

Working with Pointers
---------------------

When you work with pointers, follow security best practices listed below:

- `Handle allocation errors`_
- `Handle errors`_
- `Do not access freed memory`_
- `Avoid null pointer de-reference errors`_
- `Avoid freeing the same buffer more than once`_
- `Do not take the size of a pointer to determine the size of the object it points to`_
- `Set up a pointer to zero after freeing memory buffer`_


Handle allocation errors
++++++++++++++++++++++++

- Always check the result of a memory allocation operation.
- Explicitly handle all errors.

Handle errors
+++++++++++++

Methods that allocate memory should return ``Status`` object.

::

  services::Status initialize()
  {
          T * newData = static_cast<T *>(services::daal_malloc(_size * sizeof(T)));
          DAAL_CHECK(newData, services::ErrorMemoryAllocationFailed);
          // ...
          services::daal_free(oldData);
          return services::Status();
  }

Do not access freed memory
++++++++++++++++++++++++++

To avoid accessing memory after it is released:

- Explicitly set pointers stored in variables to ``NULL``.
- Check against ``NULL`` when in doubt.

Be careful of race conditions when freeing memory and setting the
related pointer to ``NULL`` asynchronously.

Avoid null pointer de-reference errors
++++++++++++++++++++++++++++++++++++++

To prevent reading or writing to memory with a null pointer, explicitly validate a pointer against ``NULL`` before using it.

Avoid freeing the same buffer more than once
++++++++++++++++++++++++++++++++++++++++++++

To avoid freeing buffers that have already been freed,
explicitly invalidate pointers to the buffers that have already been freed by setting pointer to ``NULL``.

Do not take the size of a pointer to determine the size of the object it points to
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This mistake may go unnoticed as the outcome might be the same on certain architectures.
Bear in mind that this might not be the case if the code is run in another environment.

Set up a pointer to zero after freeing memory buffer
++++++++++++++++++++++++++++++++++++++++++++++++++++

According to SDL, you should set a pointer to zero after using it.

::

  services::Status initialize()
  {
          T * data = static_cast<T *>(services::daal_malloc(n * sizeof(T)));
          DAAL_CHECK_MALLOC(data)
          // ...
          services::daal_free(data);
          data = NULL;
          return services::Status();
  }

Working with Buffers
--------------------

When you work with buffers, follow security best practices listed below:

- `Use a special function for copying bytes between buffers`_
- `Use a special function for memory allocation and zero-fill buffers`_
- `Check buffer size`_

Use a special function for copying bytes between buffers
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Use ``daal_memcpy_s`` to copy memory:

::

  int daal_memcpy_s(void *destination, size_t numberOfElements, const void *source, size_t count);


If its returned value is not equal to zero:

- change the library status to ``ErrorMemoryCopyFailedInternal``
- free the memory that was used for a copy
- set the buffer to ``NULL``

Use a special function for memory allocation and zero-fill buffers
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Use ``daal_calloc`` or ``service_scalable_calloc`` for memory allocation and zero-fill buffers.

.. note::

  You may use ``daal_calloc`` or ``service_scalable_calloc`` in performance-oriented cases as well,
  but make sure you initialize all data before using it.

Check buffer size
+++++++++++++++++

To prevent buffer overflow when performing multiplication operations on the size of the buffer, check its size:

::

  DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(type, multiplier1, multiplier2)

Handling exceptions
-------------------

Exceptions should be handled securely:

- Provide messages and details that contain no secure information.
- Use |short_name| recommended way for providing your exception messages.

Handling null byte injection
----------------------------

- Check input strings on null byte injection to prevent possible security vulnerabilities.
- Verify that third-party and system functions you use can appropriately handle overly-long, malformed, and non-printable symbols.

.. caution::

  If you are working with user-defined data, be careful as you might encounter null characters there.

  For example: after copying data from user-defined source into your internal buffer and inserting a null symbol in the end of it as a flag of the data ending there,
  you might encounter the user's null symbol somewhere else in the data and take is for your flag.
