/* file: assoc_rules_apriori_kernel.h */
/*******************************************************************************
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
*******************************************************************************/

/*
//++
//  Declaration of template function that computes association rules results.
//--
*/

#ifndef __ASSOC_RULES_APRIORI_KERNEL_H__
#define __ASSOC_RULES_APRIORI_KERNEL_H__

#include "src/algorithms/assocrules/assoc_rules_kernel.h"

#include "src/algorithms/assocrules/assoc_rules_apriori_itemset.i"
#include "src/algorithms/assocrules/assoc_rules_apriori_types.i"
#include "src/algorithms/assocrules/assoc_rules_apriori_tree.i"

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{
/**
 *  Structure that contains kernels for Apriori association rules mining
 */
template <typename algorithmFPType, CpuType cpu>
class AssociationRulesKernel<apriori, algorithmFPType, cpu> : public Kernel
{
public:
    /** Find "large" item sets and build association rules */
    services::Status compute(const NumericTable * a, NumericTable * r[], const daal::algorithms::Parameter * parameter);

protected:
    services::Status findLargeItemsets(size_t minSupport, size_t maxItemsetSize, assocrules_dataset<cpu> & data, ItemSetList<cpu> * L,
                                       size_t & L_size);

    Status allocateItemsetsTableData(ItemSetList<cpu> * L, size_t L_size, size_t minItemsetSize, NumericTable * largeItemsetsTable,
                                     NumericTable * largeItemsetsSupportTable, size_t & nLargeItemSets, size_t & nItemInLargeItemSets);

    Status writeItemsetsTableData(ItemSetList<cpu> * L, size_t L_size, size_t minItemsetSize, ItemsetsOrder itemsetsOrder,
                                  NumericTable & largeItemsetsTable, NumericTable & largeItemsetsSupportTable);

    Status allocateRulesTableData(NumericTable * leftItemsTable, NumericTable * rightItemsTable, NumericTable * confidenceTable, size_t nLeft,
                                  size_t nRight, size_t nRules);

    Status writeRulesTableData(AssocRule<cpu> * R, RulesOrder rulesOrder, NumericTable * leftItemsTable, NumericTable * rightItemsTable,
                               NumericTable * confidenceTable);

    /*
     *  Auxiliary methods for large item sets mining
     */

    /** Find "large" itemsets of size 1 */
    services::Status firstPass(size_t imin_s, assocrules_dataset<cpu> & data, ItemSetList<cpu> & l);

    /** Generate "large" item sets of size k+1 from "large" item sets of size k */
    hash_tree<cpu> * nextPass(size_t imin_s, size_t iset_size, assocrules_dataset<cpu> & data, ItemSetList<cpu> * L, size_t & L_size, bool & bFound,
                              hash_tree<cpu> * C_tree, services::Status & s);

    /** Test that all {n-1}-item subsets of {n}-item set are "large" item sets */
    bool pruneCandidate(size_t iset_size, const size_t * candidate, size_t * subset, hash_tree<cpu> & C_tree);

    size_t binarySearch(size_t nUniqueItems, assocRulesUniqueItem<cpu> * uniqueItems, size_t itemID);

    assocrules_itemset<cpu> * genCandidate(size_t iset_size, size_t * first_items, size_t second_item, size_t * subset_buf, hash_tree<cpu> * C_tree,
                                           services::Status & s);

    /** Generate candidate itemsets of size iset_size+1 from "large" itemsets of size iset_size */
    bool genCandidates(size_t iset_size, ItemSetList<cpu> * L, hash_tree<cpu> * C_tree, size_t nUniqueItems, assocRulesUniqueItem<cpu> * uniqueItems,
                       services::Status & s);

    /** Generate all subsets of size iset_size from a transaction and hash those subsets
        using hash tree of candidate itemsets C_tree to increment support values of candidates */
    void genSubset(size_t transactionSize, const size_t * items, size_t iset_size, size_t * subset, size_t * idx, hash_tree<cpu> & C_tree,
                   size_t & large_count);

    /** Remove candidate itemsets which support is less then minimum support
        from array of "large" item sets */
    void prune(size_t imin_s, size_t iset_size, assocrules_dataset<cpu> & data, ItemSetList<cpu> * L, hash_tree<cpu> * C_tree);

    /*
     *  Auxiliary methods for association rules discovery
     */

    /** Find item set in the list of item sets */
    const assocrules_itemset<cpu> * findItemSet(size_t items_size, const size_t * items, const ItemSetList<cpu> & L_cur);

    /** Find intersection between two item sets */
    void setIntersection(const size_t * a, size_t aSize, const size_t * b, size_t bSize, size_t * c, size_t & cSize);

    /** Find rules containing 1 item on the right */
    services::Status firstPass(double minConfidence, ItemSetList<cpu> * L, size_t itemSetSize, const size_t * items, size_t itemsSupport,
                               size_t * leftItems, AssocRule<cpu> * R, size_t & numRules, size_t & numLeft, size_t & numRight,
                               size_t & numRulesFound);

    /** Generate rules that have k+1 items on the right from the rules that have k items on the right */
    services::Status nextPass(double minConfidence, ItemSetList<cpu> * L, size_t right_size, size_t itemsSupport, size_t * leftItems,
                              AssocRule<cpu> * R, size_t & numRules, size_t & numLeft, size_t & numRight, size_t & numRulesFound, bool & found);

    /** Generate association rules from "large" item sets */
    services::Status generateRules(double minConfidence, size_t minItemsetSize, size_t L_size, ItemSetList<cpu> * L, AssocRule<cpu> * R,
                                   size_t & numRules, size_t & numLeft, size_t & numRight);

    /** Store association rules into continuous memory */
    void setRules(AssocRule<cpu> ** R, size_t numRules, int * rleft, int * rright, algorithmFPType * rconf);
};

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
