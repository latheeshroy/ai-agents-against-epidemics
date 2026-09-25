#include "mdp_solution.hpp"

namespace decPOMDP{
namespace sequential{
using namespace std;


MDP_Solution::MDP_Solution(){
}


void MDP_Solution::solve(){
    this->mdp_solver.solve();
    this->solved = true;
}


double MDP_Solution::get_value(Occupancy_State &oState){
    //-- stored in the oState ?
    if (oState.relax_value > -INFINITY)
        return oState.relax_value;    

    //-- compute
    int step = oState.get_step();
    if (step == SEARCH.horizon)
        return 0; // terminal value

    int agent = oState.get_next_agent();
    double oState_value = 0;
    if (agent == 0){
        //-- similar to simultaneous case
        const auto & belief = oState.get_belief();
        oState_value = this->mdp_solver.get_value(step, belief);
    }
    else{
        //-- recurrent greedy selection of individual actions
        //--    support -> iAction
        for (const auto & support_proba: oState.get_supports_probas(oState.is_compressed)){
            Support supp = support_proba.first;
            double proba = support_proba.second;
            oState_value += proba * this->get_value(supp, agent, step);
        }
        
    }
    
    //-- store
    oState.relax_value = oState_value;

    return oState_value;
}


/// @brief recurrent function:
/// select greedy actions for agents [agent, agent + 1, ..., PROBLEM.last_agent]
/// @param supp A support containing: i) the hidden state ii) indiv actions for agents < agent
/// @param agent The agent that take decision
double MDP_Solution::get_value(Support supp, int agent, int step) const{
    if (agent > PROBLEM.last_agent){
        int ju = supp.get_jAction();
        return this->mdp_solver.get_Qvalue(step, ju, supp.get_hiddenState());
    }

    double val = -INFINITY;
    for (int iu = 0; iu < PROBLEM.actions_number_byAgent[agent]; iu++){
        supp.set_iAction(agent, iu);
        double iu_val = this->get_value(supp, agent + 1, step); // recurrence
        val = max(val, iu_val);
    }
    return val;
}

double MDP_Solution::get_value(const POMDP::Belief & belief, Support supp, int agent, int step) const{
    if (agent > PROBLEM.last_agent){
        int ju = supp.get_jAction();
        return this->mdp_solver.get_Qvalue(step, ju, belief);
    }

    double val = -INFINITY;
    for (int iu = 0; iu < PROBLEM.actions_number_byAgent[agent]; iu++){
        supp.set_iAction(agent, iu);
        double iu_val = this->get_value(belief, supp, agent + 1, step); // recurrence
        val = max(val, iu_val);
    }
    return val;
}


}} // end namespace