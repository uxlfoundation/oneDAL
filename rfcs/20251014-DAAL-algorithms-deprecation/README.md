# Deprecation of certain DAAL algorithms in oneDAL 2025.10

## Introduction

DAAL as a project was started about 10 years ago and was initially developed
to cover broad variety of data analysis tasks.

Since then the importance of some algorithms lowered.
Having those algorithm in oneDAL doesn't add value to the product, but creates
additional maintanance costs.

The long-term goal is to cover all the functionality with oneDAL API.

Removing certain DAAL algorithms will have following positive effects
for the users of oneDAL:

- The size of the oneDAL binaries will be reduced.
- The library will be more concise and it would be easier to understand its
  possibilities.
- The build and testing time will be reduced, which can be important
  for the open source community members.

The DAAL library was launched approximately 10 years ago with the goal
of providing comprehensive coverage for a wide variety of data analysis tasks.

Over the past decade, the data science field has evolved significantly,
and the relevance of certain algorithms has diminished.

Maintaining these less-utilized algorithms in oneDAL no longer provides
meaningful value to users while continuing to impose substantial
maintenance overhead.

As part of our long-term strategy to fully transition all functionality
to the modern oneDAL API, we are deprecating select DAAL algorithms
that no longer align with current needs.

This deprecation will deliver several benefits to oneDAL users:

 - Reduced binary size: Smaller, more efficient library distributions
 - Enhanced clarity: A more focused API that makes the library's capabilities
   easier to understand and navigate
 - Improved development efficiency: Faster build and testing cycles,
   particularly benefiting open source community contributors
 - Better resource allocation: Development resources can be redirected
   toward high-impact features and optimizations

## Proposal


