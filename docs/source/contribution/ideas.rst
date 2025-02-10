.. Copyright contributors to the oneDAL project
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

Ideas for contributions
=======================

As an open-source project, we welcome community contributions to oneDAL.
This document suggests contribution directions which we consider good introductory projects with meaningful
impact. You can directly contribute to next-generation supercomputing, or just learn in depth about key 
aspects of performant machine learning code for a range of architectures. This list is expected to evolve 
with current available projects described in the latest version of the documentation.

Every project is labeled in one of three tiers based on the time commitment: 'small' (90 hours), 'medium' 
(175 hours) or 'large' (350 hours). Related topics can be combined into larger packages, though not 
completely additive due to similarity in scope (e.g. 3 'smalls' may make a 'medium' given a learning 
curve). Others may increase in difficulty as the scope increases (some 'smalls' may become large with 
in-depth C++ coding). Each idea has a description, a difficulty, and possibly an 
extended goal. They are grouped into relative similarity to allow for easy combinations.

Partial eigendecompositions for rank-deficient matrices (medium)
----------------------------------------------------------------

Some algorithms in oneDAL rely on spectral decompositions which are performed by calling corresponding
routines from LAPACK for symmetric eigenvalue calculations - particularly function ``SYEVR``. This function
is always tasked with obtaining a full eigenvalue decomposition, but being a sequential procedure, it can
also calculate partial eigencompositions - i.e. up to only the :math:`N` largest eigenvalues.

For many use-cases, components with too-small singular values in the resulting spectral decomposition are
later on discarded, in line with other LAPACK procedures such as ``GELSD``. It cannot be guaranteed without a
prior decomposition of some kind that a symmetric matrix will have a minimum number of large-enough
components, but it can be known apriori (before calling ``SYEVR``) in some cases that a minimum number of
eigenvalues should in theory be zero and thus should get discarded. In particular, a symmetric matrix
:math:`\mathbf{A}` which is the cross-product of another matrix :math:`\mathbf{X}` can have a number of
non-zero eigenvalues at most equal to the number of rows in the matrix :math:`\mathbf{X}`.

Hence, if dealing with a rank-deficient matrix :math:`\mathbf{A}` - i.e. a matrix which is the cross-product of
a matrix :math:`\mathbf{X}` which has more columns than rows - then it can be known apriori (before calling the
``SYEVR`` function) that some eigenvalues should in theory be zero or negative, and the ``SYEVR`` function call
can be sped up by not calculating a full decomposition.

This would require keeping track of the number of rows in :math:`\mathbf{X}` that generated a given matrix
:math:`\mathbf{A}` throughout the procedures that use it, up until in reaches the ``SYEVR`` function call,
which should be modified accordingly.

Algorithms that might perform eigendecompositions on rank-deficient matrices include:
    
    - Linear models.
    - PCA.
    - Precision calculation in covariance.

Extended goal: linear regression is currently implemented through a mechanism which first attemps a Cholesky
decomposition, and falls back to eigenvalue decomposition when that fails, but for rank-deficient matrices,
it will be known apriori that Cholesky will fail, so it can be skipped straight to eigendecomposition.

`(Link to github issue for discussion) <https://github.com/uxlfoundation/oneDAL/issues/3066>`__

Cholesky-based precision calculation (medium)
---------------------------------------------

In line with `scikit-learn's EmpiricalCovariance estimator <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance>`__,
the Covariance algorithm from scikit-learn-intelex by default also calculates and stores the Precision - i.e.
the inverse of the covariance. This inverse is obtained by eigendecomposition, and it may be used within the
scikit-learn-intelex interface to calculate Mahalanobis distances.

However, for full-rank matrices, it's likely faster to obtain the precision matrix out of the covariance by a
Cholesky-based inversion, at the expense of slightly reduced numerical accuracy. This could be implemented on
the oneDAL side by handling the option to calculate the precision in the C++ interface, storing the precision in
the C++ object, and calculating it with Cholesky when possible, falling back to eigenvalue-based decomposition if
Cholesky fails or is too inexact. Note that implementation of the idea about partial eigendecompositions would
also be of use here, as Cholesky-based inversion would not be applicable to rank-deficient matrices, in which case
it should go directly for eigendecomposition.

Extended goal: having a triangular factorization of the precision would also open the possibility of speeding up
Mahalanobis distance calculations, which would be faster with triangular matrices than with full-rank square root
matrices as produced by eigendecomposition. While Mahalanobis distance is typically calculated with the Cholesky of
the precision, a different Cholesky-like factorization would also suffice - for example, it would be faster to obtain
a factorization of the precision from the Cholesky of the covariance, such as suggested in
`this StackExchange answer <https://math.stackexchange.com/a/713011>`__, which could then be stored on the C++ object
and used for Mahalanobis distance calculations by adding a new method.

`(Link to github issue for discussion) <https://github.com/uxlfoundation/oneDAL/issues/3067>`__

SVD fallback from QR factorization (medium)
-------------------------------------------

While scikit-learn-intelex uses the normal-equations method for linear regression, oneDAL also offers a QR-based
algorithm, but this algorithm only works for inputs that have more rows than columns and no linear dependencies nor
constant columns. It would be ideal to make this algorithm also support non-full-rank inputs by outputting the
minimum-norm solution just like in the normal-equations method.

In theory, the QR algorithm's result could be post-processed to discard columns with linear dependencies and end up
with a solution where some coefficients are undefined (e.g. as done by the R software), but in order to match the
behavior of the normal-equations method (i.e. produce the minimum-norm solution, rather than the minimum-rank solution),
a better alternative could be to fall back to spectral decomposition done through singular value decomposition on the
original matrix :math:`\mathbf{X}` instead, in order to retain the enhanced numerical accuracy that the QR method
would have.

The idea would be to implement a fallback mechanism that would first try out QR, and if that fails, then resort to
SVD instead. Perhaps even the ``GELSD`` routine in LAPACK could be used directly. Note that QR will invariably fail when
there are more columns than rows, so in such case it should go for SVD-based procedures directly.

`(Link to github issue for discussion) <https://github.com/uxlfoundation/oneDAL/issues/3068>`__
