#include "occupancy_state_base.hpp"

namespace decPOMDP{
using namespace std;

Occupancy_State_Base::Occupancy_State_Base(): belief(PROBLEM.states_number, 0){}

Occupancy_State_Base::Occupancy_State_Base(const Vector & belief_init):
    belief(belief_init){
    for (int x = 0; x < PROBLEM.states_number; x++){
        if (belief_init[x] > 0){
            Support supp;
            supp.set_hiddenState(x);
            this->supports_proba[supp] = belief_init[x];
        }
    }
    if (SEARCH.compress)
        this->compress();
}

const Support_Val_OrderMap & Occupancy_State_Base::get_supports_probas(bool compact) const{
    if (compact){
        assert(this->is_compressed);
        return this->supports_proba_compact;
    }
    return this->supports_proba;
}

/// @brief Compute the individual histories of agent
///     and store it for computational efficiency,
///     in iHistories or iHistories_compact depending on compression choice.
/// @param compact: bool, whether using compression
/// @return A set of supports = individual histories of agent
Support_Set & Occupancy_State_Base::get_iHistories(int agent, bool compact){
    if (compact)
        assert(this->is_compressed);

    auto & vec_ihs = compact ? this->iHistories_compact : this->iHistories;

    //-- already stored ?
    if (vec_ihs.empty())
        vec_ihs.resize(PROBLEM.agents_number);
    auto & ihs = vec_ihs.at(agent);
    if (!ihs.empty())
        return ihs;

    //-- compute the set
    for (const auto & supp_proba : get_supports_probas(compact)){
        ihs.emplace(supp_proba.first.get_iHistory(agent));
    }
    return ihs;
}

int Occupancy_State_Base::get_jAction(vector<Decision_Rule> & drs, const Support & support) const{
    Support temp;
    for (int agent: PROBLEM.agents){
        int iu = this->get_iAction(drs.at(agent), support, agent);
        temp.set_iAction(agent, iu);
    }
    return temp.get_jAction();
}

int Occupancy_State_Base::get_iAction(Decision_Rule & dr, const Support & support, int agent) const{
    return dr.get_action(support.get_iHistory(agent), agent);
}

const POMDP::Belief & Occupancy_State_Base::get_belief() const{
    return this->belief;
}

/// @brief Compute the mapping: joint history -> (belief, proba).
/// Let jh be a joint history, then
/// proba = s(jh) and belief = [s(x0 | jh), s(x1 | jh), ...],
/// i.e. belief is normalized.
Belief_State & Occupancy_State_Base::get_beliefState(bool compact){
    if (compact)
        assert(this->is_compressed);

    auto & jhs_beliefProba = compact ? this->bState_compressed : this->bState;

    //-- already stored ?
    if (!jhs_beliefProba.empty())
        return jhs_beliefProba;

    //-- compute the map
    POMDP::Belief belief_null(PROBLEM.states_number, 0);
    for (const auto & support_proba: this->get_supports_probas(compact)){
        auto support = support_proba.first;
        const Support & jh = support.get_jHistory();
        int x = support.get_hiddenState();

        //-- jh already in the map ?
        auto got = jhs_beliefProba.find(jh);
        double proba = support_proba.second;
        if (got != jhs_beliefProba.end()){
            //-- yes: update
            got->second.first[x] += proba; // update the belief
            got->second.second += proba; // update the proba
        }
        else{
            //-- no: build a new belief
            POMDP::Belief new_belief = belief_null;
            new_belief[x] = proba;
            jhs_beliefProba[jh] = make_pair(new_belief, proba);       
        }
    }
    
    //-- normalize the beliefs
    for (auto & jh_beliefProba: jhs_beliefProba){
        jh_beliefProba.second.first.normalize();
    }

    return jhs_beliefProba;
}


/// @brief Compute the mapping: individual history -> (belief, proba).
/// Let ih be an individual history, then
/// proba = s(ih) and belief = [s(x0, ih), s(x1, ih), ...].
Belief_State Occupancy_State_Base::get_private_beliefState(int agent, bool compact){
    if (compact)
        assert(this->is_compressed);
    
    Belief_State private_bState;

    //-- compute the map
    POMDP::Belief belief_null(PROBLEM.states_number, 0);
    for (const auto & support_proba: this->get_supports_probas(compact)){
        auto support = support_proba.first;
        double proba = support_proba.second;
        
        const Support & ih = support.get_iHistory(agent);
        int x = support.get_hiddenState();

        //-- ih already in the map ?
        auto got = private_bState.find(ih);
        if (got != private_bState.end()){
            //-- yes: update
            got->second.first[x] += proba; // update the belief
            got->second.second += proba; // update the proba
        }
        else{
            //-- no: build a new belief
            POMDP::Belief new_belief = belief_null;
            new_belief[x] = proba;
            private_bState[ih] = make_pair(new_belief, proba);       
        }
    }
    
    //-- normalize the beliefs
    for (auto & ih_beliefProba: private_bState){
        ih_beliefProba.second.first.normalize();
    }

    return private_bState;
}


/// @brief Compute the mapping: joint histories -> {support -> proba}
unordered_map<Support, Support_Val_Map> Occupancy_State_Base::get_jhs_supportsProba(bool compact){
    unordered_map<Support, Support_Val_Map> jhs_supportsProba;
    for (const auto & support_proba: this->get_supports_probas(compact)){
        auto jh = support_proba.first.get_jHistory();
        jhs_supportsProba[jh].insert(support_proba);
    }
    return jhs_supportsProba;
}


/// @brief Compute the mapping: individual histories -> {support -> proba}
unordered_map<Support, Support_Val_Map> Occupancy_State_Base::get_private_oState(int agent, bool compact){
    unordered_map<Support, Support_Val_Map> ihs_supportsProba;
    for (const auto & support_proba: this->get_supports_probas(compact)){
        auto ih = support_proba.first.get_iHistory(agent);
        ihs_supportsProba[ih].insert(support_proba);
    }
    return ihs_supportsProba;
}


/// @brief Compute the mapping: h^{agent} -> belief_state, i.e.
/// h^{agent} -> ( jh -> (belief, proba) ), where
/// proba = s(jh) and belief = [s(x0, jh), s(x1, jh), ...].
unordered_map<Support, Belief_State> Occupancy_State_Base::get_ih_beliefState(int agent, bool compact){

    //-- we'll "split" the beliefState
    const auto & beliefState = this->get_beliefState(compact);

    //-- build the conditionnal beliefState
    unordered_map<Support, Belief_State> ih_beliefState;
    
    for (const auto & jh_beliefProba: beliefState){
        Support jh = jh_beliefProba.first; // h^{:}
        Support ih = jh.get_iHistory(agent); // h^{agent}  
        ih_beliefState[ih][jh] = jh_beliefProba.second;
    }

    return ih_beliefState;
}



/// @brief Compute the mapping: h^{agent} -> ( h^{agent+1:} -> (belief, proba) ), where
/// proba = s(h^{agent+1:}) and belief = [s(x0, h^{agent+1:}), s(x1, h^{agent+1:}), ...].
unordered_map<Support, Belief_State> Occupancy_State_Base::get_ih_beliefStateNext(int agent, bool compact){
    assert(agent < PROBLEM.last_agent);

    //-- we'll "split" the beliefState
    const auto & beliefState = this->get_beliefState(compact);

    //-- build
    unordered_map<Support, Belief_State> ih_beliefState;
    for (auto jh_beliefProba: beliefState){
        Support jh = jh_beliefProba.first; // h^{:}
        //-- "split" jh
        Support ih = jh.get_iHistory(agent); // h^{agent}
        Support ihs_nexts = jh.get_partial(agent + 1); // h^{agent+1:}
        //-- fill the mapping
        bool exists = false;
        auto got_ih = ih_beliefState.find(ih);
        if (got_ih != ih_beliefState.end()){
            auto got_nexts = got_ih->second.find(ihs_nexts); 
            if (got_nexts != got_ih->second.end()){
                exists = true;
                auto & belief = got_nexts->second.first;
                auto & proba = got_nexts->second.second;
                double proba2 = jh_beliefProba.second.second;
                belief.array() = (proba * belief.array() + proba2 * jh_beliefProba.second.first.array()) / (proba + proba2);
                proba += proba2;
            }
        }
        if (!exists){
            ih_beliefState[ih][ihs_nexts] = jh_beliefProba.second;
        }
    }

    return ih_beliefState;
}

///@brief Compare the distribution of <state, histories>,
///         on the compressed distribution if this oState and other are compressed,
///         else on (one-step) uncompressed.
bool Occupancy_State_Base::_equal(const Occupancy_State_Base & other, bool compact) const{
    static const double TOLERANCE = 1e-6;

    const auto & supports_probas = this->get_supports_probas(compact);
    const auto & supports_probas2 = other.get_supports_probas(compact);

    if (supports_probas.size() != supports_probas2.size())
        return false;

    //-- the mappings supports_proba are ordered:
    //--    lexicographic order on supports, i.e. on <state, histories>
    auto it2 = supports_probas2.begin();
    for (auto it=supports_proba.begin(); it!=supports_proba.end(); ++it){
        //-- compare supports
        if (it->first != it2->first)
            return false;
        //-- compare proba
        if (abs(it->second - it2->second) > TOLERANCE)
            return false;
        //-- next
        it2++;
    }
    return true;
}


bool Occupancy_State_Base::operator==(const Occupancy_State_Base & other) const{
    bool compact = this->is_compressed && other.is_compressed;
    return this->_equal(other, compact);
}


/// @brief Update probability for a given support:
/// - if the support is known: add the given value,
/// - else: set the given value.
/// Also update belief.
void Occupancy_State_Base::_update_proba(Support support, double proba){
    auto & supports_probas = this->supports_proba;
    auto got = supports_probas.find(support);
    if (got != supports_probas.end()){
        got->second += proba;
    }
    else{
        supports_probas.emplace(support, proba);
    }
    this->belief[support.get_hiddenState()] += proba;
}

/// @brief Normalize support probabilities to obtain a sum = 1 
void Occupancy_State_Base::normalize(){
    static const double TOLERANCE = 1e-10;
    double sum = this->get_probas_sum();
    if (abs(sum - 1) > TOLERANCE){
        for (auto it = this->supports_proba.begin(); it != this->supports_proba.end(); it++){
            it->second /= sum;
        }
    }
}

///@brief Erase the supports associated to low probabilities,
/// and normalize if some supports have been erased.
void Occupancy_State_Base::_prune(){
    static const double PRUNE_TRESHOLD = 1e-7; // low probability

    auto & supps_proba = this->supports_proba_compact; // on compressed ! 

    int n_pruned = 0;
    for (auto it = supps_proba.begin(); it != supps_proba.end(); ){
        if (it->second < PRUNE_TRESHOLD){
            it = supps_proba.erase(it); // prune
            n_pruned++;
        }
        else{
            it++;
        }
    }
    if (n_pruned > 0){
        this->normalize();
    }
}

double Occupancy_State_Base::get_probas_sum() const{
    return accumulate(this->supports_proba.begin(), this->supports_proba.end(), (double)0, 
                        [](double x, auto y){return x + y.second;});
}

int Occupancy_State_Base::get_step() const{
    return this->step;
}

int Occupancy_State_Base::size_support() const{
    return this->supports_proba.size();
}

int Occupancy_State_Base::size_support_compressed() const{
    return this->supports_proba_compact.size();
}

int Occupancy_State_Base::get_next_agent() const{
    return this->next_agent;
}

/// @brief Compute L1 distance with another occupancy state.
/// L1 distance is the sum on each support of : abs(proba(support) - other->proba(support)).
double Occupancy_State_Base::get_dist(const Occupancy_State_Base &other) const{
    static const bool USE_COMPRESS = false; // NOT COMPRESSED
    double dist = 0;
    const auto & supports_probas = this->get_supports_probas(USE_COMPRESS);
    for (const auto & support_proba: other.get_supports_probas(USE_COMPRESS)){
        auto got = supports_probas.find(support_proba.first);
        bool found = got != supports_probas.end();
        if (found){
            dist += abs(support_proba.second - got->second);
        }
        else{
            dist += support_proba.second;
        }
    }
    return dist;
}

///////////////////////////
// compression (LPE)
//////////////////////////


Support Occupancy_State_Base::get_label(const Support& support) const{
    if (!this->is_compressed)
        return support;
    auto got = this->labels.find(support); 
    assert(got != this->labels.end());
    return got->second;
}


map<Support, Support> Occupancy_State_Base::get_labels() const{
    return this->labels;
}


/// @brief Test whether the mappings supports_probas1 and supports_probas2 are similar,
///     i.e. whether they contain the same pairs (support, value)
///     with a tolerance fixed by SEARCH.compress_threshold
bool Occupancy_State_Base::_test_iHistories_Equivalence(const Support_Val_OrderMap& supports_probas1, const Support_Val_OrderMap& supports_probas2) const {    
    for(auto& support_proba: supports_probas1){
        double proba1 = support_proba.second;
        
        auto got = supports_probas2.find(support_proba.first);
        double proba2 = got != supports_probas2.end() ? got->second: 0;
        
        if(abs(proba1 - proba2) > SEARCH.compress_threshold)
            return false;
    }
    return true;
}


/// @brief LPE compression.
/// Modify:
///     this->supports_proba_compact (compression of this->supports_proba)
///     this->labels (this->labels[supp] = label foreach supp in this->supports_proba)
///     this->is_compressed (= true)
void Occupancy_State_Base::compress(){
    assert(!this->is_compressed || (this->get_step() == 0 && this->get_next_agent() == 0));

    if (this->next_agent != 0){
        //-- sequential case, intermediate agents: done in do_step()
        this->is_compressed = true;        
        return;
    }

    //-- equivalences store correspondances iHist -> support (foreach agent)
    vector<map<Support, Support>> equivalences(PROBLEM.agents_number);

    auto to_compress = this->get_supports_probas(false);

    for (int agent: PROBLEM.agents){

        //-- compute the (ordered) mapping:
        //--    iHist ->  <proba iHist, map: iMasked_supp -> proba> (iMasked_supp being the support without the agent iHist)
        auto results = this->_get_iHistory_supports_map(agent, to_compress);

        //-- compute equivalence classes
        //--    for an agent, a class is the mapping: iHist -> label 
        //--    (where label is the lower iHist in the class following lexicographic ordering)
        for(auto res1= results.begin(); res1 != results.end(); res1++){ //-- foreach iHist
            const Support_Val_OrderMap & supports_probas1 = res1->second.second;
            for(auto res2 = std::next(res1); res2 != results.end();){ //-- foreach iHist2 > iHist
                const auto & supports_probas2 = res2->second.second;
                if(_test_iHistories_Equivalence(supports_probas1, supports_probas2)){
                    //-- update equivalence class
                    equivalences.at(agent)[res2->first] = res1->first; // iHist is the label of iHist2
                    res1->second.first += res2->second.first; // add iHist2 proba to the label proba
                    res2 = results.erase(res2); // erase iHist2
                }
                else{
                    res2++;
                }
            }
        }

        //-- update compact supports_proba
        to_compress.clear();
        for (auto res: results){ // res = <iHist, <iHist proba, map: imasked support -> normalized proba>>>
            Support iHist = res.first;
            double iHist_proba = res.second.first;
            for(auto & support_proba : res.second.second){
                Support support = support_proba.first;
                double proba = iHist_proba * support_proba.second;
                support.set_iHistory(agent, iHist);

                auto got = to_compress.find(support);
                if (got != to_compress.end()){
                    got->second += proba;
                }
                else{
                    to_compress.emplace(support, proba);
                }
            }
        }
    }

    this->supports_proba_compact = to_compress;

    //-- finalize
    //--    memoize labels
    for (const auto& support_proba: this->get_supports_probas(false)){
        Support supp = support_proba.first;
        Support label = supp;
        for(int agent = 0; agent < PROBLEM.agents_number; agent++){
            Support iHistory =  supp.get_iHistory(agent);
            auto got = equivalences.at(agent).find(iHistory);
            if(got != equivalences.at(agent).end()){
                label.set_iHistory(agent, got->second);
            }
        }

        this->labels[supp] = label;
    }

    // this->_prune(); cout << "warning: prune (compress)" << endl; ////////////

    this->is_compressed = true;
}


/// @brief Compression utility.
///     Compute the mapping: iHist -> pair<double, map<Support, double>>
///        where pair<double, map<Support, double>> correspond to <proba iHist, map: iMasked_supp -> proba>
///        (iMasked_supp being the support without the agent iHist, i.e. masked).
map<Support, pair<double, Support_Val_OrderMap>> Occupancy_State_Base::_get_iHistory_supports_map(int agent, const Support_Val_OrderMap & to_compress) const{
    map<Support, pair<double, Support_Val_OrderMap>> results;
    
    //-- fill the mapping
    for (const auto & support_proba: to_compress){
        Support iHist = support_proba.first.get_iHistory(agent);
        Support iMasked_supp = support_proba.first.get_iHist_masked(agent);
        double iMasked_proba = support_proba.second;

        //-- update
        if (results.find(iHist) == results.end()){
            Support_Val_OrderMap iMasked = {{iMasked_supp, iMasked_proba}};
            auto p = make_pair(iMasked_proba, iMasked);
            results.emplace(iHist, p);
        }
        else{
            results[iHist].first += iMasked_proba; // ihist total proba
            results[iHist].second.emplace(iMasked_supp, iMasked_proba);
        }
    }

    //-- normalize probas
    //--    i.e. for a given iHist, the sum of iMasked probas equals 1
    for (auto& result: results){
        double proba = result.second.first;
        for (auto & iMaskedSupp_proba: result.second.second)
            results[result.first].second[iMaskedSupp_proba.first] /= proba;
    }

    return results;
}

}// end namespace