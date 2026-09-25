#pragma once

#include <fstream>
#include <sstream>
#include <cstdint>
#include <map>
#include "../core/_module.hpp"
#include "occupancy_state.hpp"
#include "value_representation/Q_representation.hpp"
#include "value_representation/mdp_solution.hpp"
#include "value_representation/reward_shaping.hpp"

namespace decPOMDP{
namespace sequential{
    using namespace std;


class oSarsa{
    int horizon_tau; // sequential horizon
    Occupancy_State oState0; // initial oState (from initial belief) — default / fallback
    vector<Occupancy_State> oStates0; // multiple initial oStates (multi-belief mode)
    int current_belief_idx = 0; // round-robin index into oStates0

    //-- policy
    Policy policy;
    vector<Qvalues_Tabular> Qfunctions; // Qfunctions[tau]: associated to policy suffix
    int improved_lastTau = -1;

    //-- epsilon-greedy
    double epsilon;
    bool greedy;
    double rnd_0_1 = 1.0; // for portfolio choice
    MDP_Solution mdp_sol;
    vector<int> blind_actions;

    //-- simulated annealing
    double temperature;
    bool improved;
    double final_eval = -INFINITY;
    Policy final_policy;
    
    //-- reward shaping    
    Reward_Shaping rew_shaping;

    //-- reachables
    vector<Support_Set> reachable_sets; // reachable_sets[tau]: sets of reachables <x,o,u^>

    //-- streamed backup execution metrics
    vector<uint64_t> Qversions;
    uint64_t q_updates = 0;
    uint64_t staleness_count = 0;
    uint64_t staleness_sum = 0;
    uint64_t staleness_max = 0;
    uint64_t stage_publications = 0;
    map<uint64_t, uint64_t> staleness_histogram;
    ofstream async_log;
    ofstream metrics_log;
    ofstream phase_log;
    ofstream execution_trace;

    utils::Logger logger;

public:
    oSarsa();

    void solve();

protected:
    //-- oSarsa main functions

    void _init_policy();
    void _update_policy(); // policy improvement
    void _update_Q(); // backup operator
    void _update_Q_parallel();
    void _update_q(Support, Policy &, int tau);
    double _estimate_q(Support, Policy &, int tau, const Qvalues_Tabular * q_func_next) const;
    double _estimate_q_lazy(
        Support,
        Policy &,
        int tau,
        const vector<Qvalues_Tabular> & snapshot,
        vector<Support_Val_Map> & memo,
        vector<tuple<int, Support, double>> & entries) const;
    void _ensure_Q(Occupancy_State &, Policy &);
    void _ensure_initial_Q();
    void _invalidate_lazy_Q(int last_tau = -1);
    

    //-- epsilon greedy

    Decision_Rule _select_epsilon_greedy(Occupancy_State &);
    Decision_Rule _select_greedy(Occupancy_State &);
    Decision_Rule _select_random(Occupancy_State &);
    Decision_Rule _select_MDP(Occupancy_State &);
    Decision_Rule _select_blind(Occupancy_State &);
    double _get_epsilon(int iter) const;

    //-- simulated annealing

    bool _accept_decrease(double decrease, int agent) const;
    double _get_temperature(double epsilon) const;

    //-- multi-belief

    void _load_beliefs(); // load beliefs from file into oStates0
    Occupancy_State & _get_current_oState0(); // round-robin current belief
    void _advance_belief_idx(); // move to next belief

    //-- utils

    double _get_exactValue(Occupancy_State oState, Policy & pi) const;
    double _get_Qvalue(Occupancy_State &, Decision_Rule &) const;
    void _set_reachables();
    void _extend_jointDecisionRule_LPE(Decision_Rule &, Occupancy_State &) const;
    void _log(int iter);
};


}}