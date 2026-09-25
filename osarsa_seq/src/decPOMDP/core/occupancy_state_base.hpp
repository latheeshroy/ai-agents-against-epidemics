#pragma once

#include "../../POMDP/belief.hpp"
#include "support.hpp"
#include "decision_rule.hpp"

#include <iostream>
#include <unordered_map>
#include <map>
#include <initializer_list>

namespace decPOMDP{
using namespace std;


using Belief_State = unordered_map<Support, pair<POMDP::Belief, double>>;


class Occupancy_State_Base{
protected:
    int next_agent = 0; // next agent to play
    int step = 0;  // time-step
    Support_Val_OrderMap supports_proba; // main data
    Support_Val_OrderMap supports_proba_compact; // main data (compressed)
    //-- memoization
    POMDP::Belief belief;
    Belief_State bState;
    Belief_State bState_compressed;
    vector<unordered_set<Support>> iHistories;
    vector<unordered_set<Support>> iHistories_compact;
    //-- compression
    map<Support, Support> labels;

public:
    bool is_compressed = false;
    double relax_value = -INFINITY; // relaxation value (from mdp/pomdp solution)

    //-- constructors
    Occupancy_State_Base();
    Occupancy_State_Base(const Vector & belief_init); // for root occupancy state
    virtual void test() = 0;
    virtual ~Occupancy_State_Base(){}
    
    //-- get methods
    const Support_Val_OrderMap & get_supports_probas(bool compact) const;
    int get_jAction(vector<Decision_Rule> &, const Support &) const;
    int get_iAction(Decision_Rule &, const Support &, int agent) const;
    const POMDP::Belief & get_belief() const;
    Belief_State & get_beliefState(bool compact);
    Belief_State get_private_beliefState(int agent, bool compact);
    unordered_map<Support, Belief_State> get_ih_beliefState(int agent, bool compact);
    unordered_map<Support, Belief_State> get_ih_beliefStateNext(int agent, bool compact);
    unordered_map<Support, unordered_map<Support, double>> get_jhs_supportsProba(bool compact);
    unordered_map<Support, unordered_map<Support, double>> get_private_oState(int agent, bool compact);
    Support_Set & get_iHistories(int agent, bool compact);
    int get_step() const;
    double get_dist(const Occupancy_State_Base &other) const;
    int size_support() const;
    int size_support_compressed() const;
    int get_next_agent() const;

    //-- normalization
    void normalize();
    double get_probas_sum()const;

    //-- compression
    void compress();
    Support get_label(const Support&) const;
    map<Support, Support> get_labels() const;

    //-- operators
    bool _equal(const Occupancy_State_Base & other, bool compact) const;
    bool operator==(const Occupancy_State_Base & other) const;

protected:
    void _update_proba(Support, double);
    void _prune();
    
    //-- compression
    map<Support, pair<double, map<Support, double>>> _get_iHistory_supports_map(int agent, const Support_Val_OrderMap &) const;
    bool _test_iHistories_Equivalence(const Support_Val_OrderMap& arg1, const Support_Val_OrderMap& arg2) const;
};


struct oStateHash{
    static constexpr double TOLERANCE = 1e-6;
    static constexpr double approx_factor = 1 / TOLERANCE;

    static size_t get_hash(const Occupancy_State_Base & oState, bool compact){
        size_t seed = 0;
        for (const auto & supp_proba: oState.get_supports_probas(compact)){
            utils::hash_combine(seed, supp_proba.first.get_container());
            int approx_proba = int(supp_proba.second * approx_factor); // rounding
            utils::hash_combine(seed, approx_proba);
        }
        return seed;
    }

    size_t operator()(const Occupancy_State_Base & oState) const {
        return this->get_hash(oState, oState.is_compressed);
    }
};


struct oStateHash_Uncompressed{

    size_t operator()(const Occupancy_State_Base & oState) const {
        return oStateHash::get_hash(oState, false);
    }
};


struct oState_Equal_Uncompressed{

    bool operator()(const Occupancy_State_Base & a, const Occupancy_State_Base & b) const{
        bool compact = false;
        return a._equal(b, compact);
    }
};


}// end namespace