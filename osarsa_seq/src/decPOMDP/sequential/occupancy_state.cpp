#include "occupancy_state.hpp"

#include <iostream>

namespace decPOMDP{
namespace sequential{
using namespace std;

Occupancy_State::Occupancy_State(): Occupancy_State_Base(){}
Occupancy_State::Occupancy_State(const Vector & belief_init): Occupancy_State_Base(belief_init){}
Occupancy_State::Occupancy_State(int step, int next_agent): 
    Occupancy_State_Base(){
    this->step = step;
    this->next_agent = next_agent;
}


/// @brief Compute the next occupancy state.
/// @param decision_rule The agent decision rule.
Occupancy_State Occupancy_State::do_step(Decision_Rule &decision_rule) const{
    int next_agent = this->next_agent == PROBLEM.last_agent ? 0 : this->next_agent + 1;
    int next_step = this->next_agent == PROBLEM.last_agent ? this->step + 1 : this->step;
    Occupancy_State oState_next(next_step, next_agent);

    if (this->next_agent != PROBLEM.last_agent){
        oState_next.belief = this->belief; // intermediate agent: we only insert her actions in the supports
        int agent = this->next_agent;
        //-- update uncompressed support, i.e. supports_proba
        for (const auto & support_proba: this->supports_proba){
            auto support = support_proba.first;
            //-- find action
            auto label = this->get_label(support);
            int action = decision_rule.get_action(label.get_iHistory(agent), agent);
            //-- update
            support.set_iAction(agent, action);
            oState_next.supports_proba.emplace(support, support_proba.second);
        }
        //-- update compressed support, i.e. supports_proba_compact
        for (const auto & support_proba: this->supports_proba_compact){
            auto support = support_proba.first;
            int action = decision_rule.get_action(support.get_iHistory(agent), agent);
            support.set_iAction(agent, action);
            oState_next.supports_proba_compact.emplace(support, support_proba.second);
        }
        //-- update labels, i.e. labels
        for (const auto & supp_label: this->labels){
            auto support = supp_label.first;
            auto label = supp_label.second;
            int action = decision_rule.get_action(label.get_iHistory(agent), agent);
            label.set_iAction(agent, action);
            support.set_iAction(agent, action);
            oState_next.labels.emplace(support, label);
        }
    }
    else{
        for (auto & support_proba: this->get_supports_probas(this->is_compressed)){
            auto support = support_proba.first;
            double proba = support_proba.second;
            //-- individual action
            int action = this->get_iAction(decision_rule, support, this->next_agent);
            support.set_iAction(this->next_agent, action);
            //-- new supports: reachables from the (hidden_state, action_joint)
            const auto &reachables = PROBLEM.get_reachables(support.get_hiddenState(), support.get_jAction());
            for (const auto & reached: reachables){
                auto next_support = support.do_step(reached.y, reached.joint_obs);
                oState_next._update_proba(next_support, proba * reached.proba);
            }
        }
    }

    return oState_next;
}

vector<Occupancy_State> Occupancy_State::do_step(vector<Decision_Rule> &decision_rules)const{
    vector<Occupancy_State> oStates;
    oStates.reserve(PROBLEM.agents_number);
    const Occupancy_State * oState = this;
    for (int agent: PROBLEM.agents){
         oStates.push_back(oState->do_step(decision_rules.at(agent)));
         oState = &oStates.at(agent);
    }
    return oStates;
}

/// @brief Compute expected one-step reward, given (next-agent) individual decision-rule.
double Occupancy_State::get_expected_reward(Decision_Rule & decision_rule) const{        
    if (this->next_agent != PROBLEM.last_agent)
        return 0;
    double reward = 0;
    for (auto & supp_proba: this->get_supports_probas(this->is_compressed)){
        //-- find jaction from: i) decision_rule (for this agent) ii) supp_proba (former agent) 
        auto supp = supp_proba.first;
        int action = this->get_iAction(decision_rule, supp, this->next_agent); 
        supp.set_iAction(this->next_agent, action);
        int action_joint = supp.get_jAction();
        //-- reward
        reward += supp_proba.second * PROBLEM.rewards_matrix(supp.get_hiddenState(), action_joint);
    }
    return reward;
}


int Occupancy_State::get_tau(int step, int agent){
    return step * PROBLEM.agents_number + agent;
}

pair<int, int> Occupancy_State::get_step_agent(int tau){
    return make_pair(tau / PROBLEM.agents_number, tau % PROBLEM.agents_number);
}

int Occupancy_State::get_tau() const{
    return  Occupancy_State::get_tau(this->step, this->next_agent);
}


/// @brief Compute transitions probabilities for p(s,a):
///        Pr(supp' | s, iu, ih) * s(ih), i.e. for a(iu |ih) == 1
/// @return A map: supp' -> (map: <ih, iu> -> proba)
unordered_map<Support, Support_Val_Map> Occupancy_State::get_transitions() const{
    int agent = this->get_next_agent();

    //-- compute transitions probabilities (utility for ratio computation):
    //--    Pr(supp' | s, iu, ih) * s(ih), i.e. for a(iu |ih) == 1
    unordered_map<Support, Support_Val_Map> transitions; // transitions[supp'][<ih,iu>]
    for (const auto & supp_proba: this->get_supports_probas(this->is_compressed)){
        Support supp = supp_proba.first;
        double proba = supp_proba.second;
        for (int iu = 0; iu < PROBLEM.actions_number_byAgent[agent]; iu++){
            auto ihiu = supp.get_iHistory(agent);
            ihiu.set_iAction(agent, iu); // <ih, iu>
            supp.set_iAction(agent, iu); // <supp, iu>
            if (agent == PROBLEM.last_agent){
                const auto &reachables = PROBLEM.get_reachables(supp.get_hiddenState(), supp.get_jAction());
                for (const auto & reached: reachables){
                    //-- proba = s(x,u^{:agent-1},jh) Pr()
                    //-- reached.proba = Pr(y,z | x,ju) ???
                    auto next_support = supp.do_step(reached.y, reached.joint_obs);
                    auto got_next = transitions.find(next_support);
                    if (got_next != transitions.end() && got_next->second.find(ihiu) != got_next->second.end())
                        transitions[next_support][ihiu] += proba * reached.proba;
                    else
                        transitions[next_support][ihiu] = proba * reached.proba;
                }
            }
            else{
                auto next_support = supp;
                auto got_next = transitions.find(next_support);
                if (got_next != transitions.end() && got_next->second.find(ihiu) != got_next->second.end())
                    transitions[next_support][ihiu] += proba;
                else
                    transitions[next_support][ihiu] = proba;
            }
        }
    }
    
    return transitions;
}


/// @brief check whether this contains an unreachable support
bool Occupancy_State::allReachable(const unordered_map<Support, Support_Val_Map> & transitions, bool compact) const{
    compact = compact && this->is_compressed;
    auto const & supps_proba = this->get_supports_probas(compact);
    for (const auto & supp_proba: supps_proba){
        if (transitions.find(supp_proba.first) == transitions.end())
            return false;
    }
    return true;
}


ostream &operator<<(ostream &os, const Occupancy_State &s){
    const auto & supports_proba = s.get_supports_probas(s.is_compressed);
    os << "step " << s.step
        << " next agent " << s.next_agent
        << " compressed ?" << (s.is_compressed ? " YES": " NO") 
        << " size " << supports_proba.size()
        << endl;
    for (const auto & support_proba: supports_proba){
        os << support_proba.first.get_repr(s.step, s.next_agent) << "  proba: " <<  support_proba.second << endl;
    }
    return os;
}


string Occupancy_State::get_repr_short() const{
    ostringstream os;
    os << "step " << step
        << " next agent " << next_agent
        << " compressed ?" << (is_compressed ? " YES": " NO") 
        << " size_supp " << size_support();
    if (is_compressed)
        os << " size_compact " << size_support_compressed();

    int cpt1=0, cpt2=0, cpt3=0, cpt4=0, cpt5=0;
    for (const auto & support_proba: get_supports_probas(is_compressed)){
        double proba = support_proba.second;
        if (proba > 1e-1) cpt1++;
        else if (proba > 1e-2) cpt2++;
        else if (proba > 1e-3) cpt3++;
        else if (proba > 1e-4) cpt4++;
        else cpt5++;
    }
    os << " >1e-1 " << cpt1
        << " >1e-2 " << cpt2
        << " >1e-3 " << cpt3
        << " >1e-4 " << cpt4
        << " <1e-4 " << cpt5;
    return os.str();
}

////////////////////////////////
// Occupancy_State_Corner:
// An occupancy state with only one support.
////////////////////////////////

Occupancy_State_Corner::Occupancy_State_Corner(int step, int next_agent, const Support & support): 
    Occupancy_State(step, next_agent){
    this->_update_proba(support, 1.0);
    if (SEARCH.compress){
        this->supports_proba_compact = this->supports_proba;
        this->is_compressed = true;
    }
}

Occupancy_State_Corner::Occupancy_State_Corner(const Occupancy_State & oState, const Support & support):
    Occupancy_State_Corner(oState.get_step(), oState.get_next_agent(), support){
}


}} // end namespace