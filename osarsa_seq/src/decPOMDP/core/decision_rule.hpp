#pragma once

#include "support.hpp" 

#include <iostream> 
#include <unordered_map>

namespace decPOMDP{
using namespace std;


enum RuleType{
    REGULAR,
    RANDOM, // sample an individual policy in the determinist policy space (i.e. for a given individual history, always select the same action)
    BLIND_DR // constant action (regardless the individual history) 
};
inline int ACTION_NOT_FOUND = -1;


/// @brief A decision rule map supports to action.
struct Decision_Rule{
    RuleType type = RuleType::REGULAR;
    Support_Int_Map iHist_action_map; // map individual histories (represented by a support) to individual actions (an integer)
    int blind_action = -1;
    int default_action = ACTION_NOT_FOUND;

    Decision_Rule();
    Decision_Rule(RuleType);
    Decision_Rule(int constant_action);
    
    int get_action(const Support & iHist, int agent = -1);
    void set_action(const Support & iHist, int action);
    size_t size() const;
    bool operator==(const Decision_Rule & other) const;

    static pair<int, double> worst_heuristic(); // return <the action, a lower bound for the reward>
    static pair<int, double> blind_heuristic(); // return <the action, a lower bound for the reward>
    static pair<int, double> myopic_heuristic(int agent, Support); // return <the action, a lower bound for the reward>
    static pair<int, double> lucky_heuristic(); // return <the action, an higher bound for the reward>

    friend ostream &operator<<(ostream &os, const Decision_Rule &d);
    string get_repr(int agent, int step) const;
};


//------------------------------------
//-- policy
//------------------------------------

/// @brief A joint decision rule is a collection of Decision_Rule (for each agent).
struct Joint_DecisionRule{
    vector<Decision_Rule> decision_rules; // for each agent

    Joint_DecisionRule();
    Joint_DecisionRule(int nb_agents);
    Joint_DecisionRule(const vector<Decision_Rule> &);

    int get_iAction(int agent, const Support & ih);
    int get_jAction(const Support & supp);
    void set_iAction(int agent, const Support & ih, int action);
    void set_jAction(const Support & supp, int action);
    void set_default_actions(vector<int> actions);

    friend ostream &operator<<(ostream &os, const Joint_DecisionRule & jdr);
    string get_repr(int step) const;
};


/// @brief A policy is composed by
/// a decision rule for each agent and for each time-step.    
class Policy{
    vector<Joint_DecisionRule> joint_drs; // for each time-step

public:
    Policy(int horizon, int nb_agents);

    int get_iAction(int step, int agent, const Support & ih);
    int get_jAction(int step, const Support & supp);

    Decision_Rule & get_decisionRule(int step, int agent);
    Joint_DecisionRule & get_joint_decisionRule(int step); 
    
    void set_decisionRule(int step, int agent, const Decision_Rule &);

    friend ostream &operator<<(ostream &os, const Policy & pi);
};


//------------------------------------
//-- enumeration
//------------------------------------

using Variation = vector<int>;
using Variations = vector<Variation>;

/// @brief A variation is a vector of actions.
class Variations_Generator{
    static const bool STORE = true;

public:
    vector<Decision_Rule> get_decision_rules(const unordered_set<Support> &, int actions_number);
    void clear();

private:
    Variations variations;
    int length;
    int actions_number;
    int num_variation;
    
    //-- memoize
    unordered_map<int, unordered_map<int, Variations>> variations_map; // variations[length][actions_number]

    void _generate(int depth, const Variation &);
    Variations& _get_variations(int length, int actions_number);

}inline VARIATIONS_GENERATOR;


/// @brief Compute all decision rules for a given oState
class Decision_Rules_Enumeration{
    int agents_number;
    int agents_num = 0;
    vector<int> size_byAgent; // decision rules number by agent
    vector<vector<Decision_Rule>> decision_rules;

public:
    int size = 0; // total joint decision rules number

    Decision_Rules_Enumeration(int agents_number);
    void add_agent(const unordered_set<Support> &, int agent_actions_number);
    vector<Decision_Rule> get(int index);
    Joint_DecisionRule get_joint_drs(int index);
};

}