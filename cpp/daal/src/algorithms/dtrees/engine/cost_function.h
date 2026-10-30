/* file: cost_function.h */
/*******************************************************************************
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
*******************************************************************************/

/*
//++
//  Split cost functions of the tree engine: Gini for classification, MSE for
//  regression (design 4.3).
//
//  Both score a split by a proxy P = sum over children of (child statistic)^2 / child weight
//  (Gini: per-class weights; MSE: weighted response sums). With the parent's proxy P0, the
//  weighted impurity decrease of the split is P - P0, so one number ranks candidates and gives the
//  decrease that min_impurity_decrease and MDI importance use.
//--
*/

#ifndef __DTREES_ENGINE_COST_FUNCTION_H__
#define __DTREES_ENGINE_COST_FUNCTION_H__

#include <cstddef>
#include <cstdint>

#include "src/algorithms/dtrees/engine/response_set.h"
#include "src/services/service_defines.h"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace engine
{
namespace internal
{
using daal::internal::CpuType;

// Per-record view used by the cost functions: response (class or value), weight, and the bootstrap
// count kept for node sample counts (1 for weighted layouts, which carry count * user weight).
inline uint32_t recordClass(const ClsCountRecord & r)
{
    return r.cls;
}
template <typename algorithmFPType>
inline uint32_t recordClass(const ClsWeightRecord<algorithmFPType> & r)
{
    return r.cls;
}
template <typename algorithmFPType>
inline double recordResponse(const RegCountRecord<algorithmFPType> & r)
{
    return r.y;
}
template <typename algorithmFPType>
inline double recordResponse(const RegWeightRecord<algorithmFPType> & r)
{
    return r.y;
}
inline uint32_t recordCount(const ClsCountRecord & r)
{
    return r.count;
}
template <typename algorithmFPType>
inline uint32_t recordCount(const RegCountRecord<algorithmFPType> & r)
{
    return r.count;
}
template <typename algorithmFPType>
inline uint32_t recordCount(const ClsWeightRecord<algorithmFPType> &)
{
    return 1;
}
template <typename algorithmFPType>
inline uint32_t recordCount(const RegWeightRecord<algorithmFPType> &)
{
    return 1;
}

/**
 *  \brief Gini impurity over nClasses classes
 *
 *  A side's state is its per-class weights, its weight, and the sum of squared class weights, kept
 *  incrementally so adding a row is O(1).
 *
 *  The right side's sum starts at the parent's and only shrinks, so with fractional weights its rounding
 *  error, which grows with the number of updates, would come to dominate it near the end of a scan.
 *  It is therefore summed again from the class weights whenever the right weight has halved since the
 *  last exact sum: O(nClasses) a logarithmic number of times per scan, and an error relative to the
 *  right side throughout. Count layouts sum integers exactly, so the resync does not change their splits.
 */
template <CpuType cpu>
class GiniCost
{
public:
    static constexpr bool classification = true;

    /// counts holds 2 * nClasses doubles of scratch: the left side, then the parent.
    GiniCost(size_t nClasses, double * counts) : _nClasses(nClasses), _left(counts), _parent(counts + nClasses) {}

    /// Sets the parent row by row: startParent, addParent per row, finishParent.
    void startParent()
    {
        for (size_t c = 0; c < _nClasses; ++c) _parent[c] = 0;
        _w = 0;
    }
    void addParent(uint32_t c, double a)
    {
        _parent[c] += a;
        _w += a;
    }
    void finishParent()
    {
        _sq = 0;
        for (size_t c = 0; c < _nClasses; ++c) _sq += _parent[c] * _parent[c];
    }

    /// Sets the parent from labels and weights of its n rows.
    void setParent(const uint32_t * label, const double * w, size_t n)
    {
        startParent();
        for (size_t i = 0; i < n; ++i) addParent(label[i], w[i]);
        finishParent();
    }

    double weight() const { return _w; }
    /// sum_c w_c^2 / w.
    double parentProxy() const { return _w > 0 ? _sq / _w : 0; }
    double impurity() const { return _w > 0 ? clip(1 - _sq / (_w * _w)) : 0; }
    const double * classWeights() const { return _parent; }

    void resetLeft()
    {
        for (size_t c = 0; c < _nClasses; ++c) _left[c] = 0;
        _wL     = 0;
        _sqL    = 0;
        _sqR    = _sq;
        _wRSync = _w;
    }

    /// Moves a row of class c and weight a from the right side to the left.
    void addLeft(uint32_t c, double a)
    {
        const double r = _parent[c] - _left[c];
        _sqL += a * (2 * _left[c] + a);
        _sqR += a * (a - 2 * r);
        _left[c] += a;
        _wL += a;
        if (2 * (_w - _wL) < _wRSync) syncRight();
    }

    double leftWeight() const { return _wL; }
    double rightWeight() const { return _w - _wL; }

    /// sum_c L_c^2 / wL + sum_c R_c^2 / wR for the current left side; both sides must have weight.
    double proxy() const { return _sqL / _wL + clip(_sqR) / (_w - _wL); }

private:
    static double clip(double v) { return v > 0 ? v : 0; }

    /// The right side's sum of squared class weights, from the class weights.
    void syncRight()
    {
        _sqR = 0;
        for (size_t c = 0; c < _nClasses; ++c)
        {
            const double r = _parent[c] - _left[c];
            _sqR += r * r;
        }
        _wRSync = _w - _wL;
    }

    size_t _nClasses;
    double * _left;
    double * _parent;
    double _w      = 0;
    double _sq     = 0;
    double _wL     = 0;
    double _sqL    = 0;
    double _sqR    = 0;
    double _wRSync = 0; // right weight at the last exact sum
};

/**
 *  \brief Mean squared error
 *
 *  A side's state is its weight and weighted response sum, in double whatever algorithmFPType is.
 *  Raw sums would cancel as (offset / spread)^2 for a node whose responses sit far from zero against their
 *  spread, so both the split proxy and the impurity are centered:
 *  - The proxy centers each side's sum on the parent's mean. That moves every candidate's proxy and the
 *    parent's by the same constant, so the chosen split and its decrease do not change, and the scan's
 *    rounding grows only as offset / spread.
 *  - The impurity sums squared deviations from a center given before the pass (setCenter), the node's mean
 *    as its parent's split sums estimate it, so the node's records are read once, and subtracts the squared
 *    distance from the center to the node's mean. That mean's rounding grows with the offset, but the
 *    center comes from sums of the same records, so the distance, and the correction's error, stay small.
 */
template <CpuType cpu>
class MseCost
{
public:
    static constexpr bool classification = false;

    /// The center of the next parent's deviations; 0 until set.
    void setCenter(double c) { _center = c; }
    void startParent()
    {
        _w   = 0;
        _s   = 0;
        _dev = 0;
    }
    void addParent(double y, double a)
    {
        const double d = y - _center;
        _w += a;
        _s += a * y;
        _dev += a * d * d;
    }
    void finishParent() { _c = _w > 0 ? _s / _w : 0; }

    void setParent(const double * y, const double * w, size_t n)
    {
        startParent();
        for (size_t i = 0; i < n; ++i) addParent(y[i], w[i]);
        finishParent();
        setCenter(_c);
        startParent();
        for (size_t i = 0; i < n; ++i) addParent(y[i], w[i]);
        finishParent();
    }

    double weight() const { return _w; }
    double sum() const { return _s; }
    double parentProxy() const
    {
        const double s = _s - _c * _w;
        return _w > 0 ? s * s / _w : 0;
    }
    double impurity() const
    {
        if (!(_w > 0)) return 0;
        const double off = _c - _center;
        const double v   = _dev / _w - off * off;
        return v > 0 ? v : 0;
    }
    double mean() const { return _c; }

    void resetLeft()
    {
        _wL = 0;
        _sL = 0;
    }
    void addLeft(double y, double a) { addLeftSums(a, a * y); }
    /// Moves rows of total weight a and weighted response sum s to the left.
    void addLeftSums(double a, double s)
    {
        _wL += a;
        _sL += s;
    }
    double leftWeight() const { return _wL; }
    double rightWeight() const { return _w - _wL; }
    double leftSum() const { return _sL; }
    double proxy() const
    {
        const double wR = _w - _wL;
        const double sL = _sL - _c * _wL;
        const double sR = (_s - _sL) - _c * wR;
        return sL * sL / _wL + sR * sR / wR;
    }

private:
    double _w      = 0;
    double _s      = 0;
    double _c      = 0; // the parent's mean
    double _center = 0; // the impurity pass's center
    double _dev    = 0; // the parent's weighted sum of squared deviations from _center
    double _wL     = 0;
    double _sL     = 0;
};

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
