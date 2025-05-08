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

::

  class DotReducer : public daal::Reducer
  {
  public:
    float dot;

    DotReducer(size_t n, const float * a, const float * b) : dot(0.0f), n_(n), a_(a), b_(b) {}

    virtual ReducerUniquePtr create() const override
    {
      // Memory for the local result will be freed automatically
      // when the local result is no longer needed
      return daal::internal::makeUnique<DotReducer, DAAL_BASE_CPU>(n_, a_, b_);
    }

    virtual void update(size_t begin, size_t end) override
    {
      float localDot(0.0f);
      for (size_t i = begin; i < end; ++i)
      {
        localDot += a_[i] * b_[i];
      }
      dot += localDot;
    }

    virtual void join(Reducer * otherReducer) override
    {
      DotReducer * other = dynamic_cast<DotReducer *>(otherReducer);
      dot += other->dot;
    }

  private:
    size_t n_;
    const float * a_;
    const float * b_;
  };
