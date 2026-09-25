#include "reward_shaping.hpp"

namespace decPOMDP{
namespace sequential{
    using namespace std;


/// @brief Constructor.
/// Fill rewards_matrices (a matrix by agent) that replace PROBLEM.rewards_matrix.
Reward_Shaping::Reward_Shaping(){
    if (!SEARCH.use_reward_shaping)
        return;
    cout << "$$$$$ Reward_Shaping : init $$$$$" << endl;

    //-- init matrices
    this->rewards_matrices.resize(PROBLEM.agents_number,
        Matrix<double>(PROBLEM.states_number, PROBLEM.actions_joint_number, -INFINITY));
    
    //-- fill matrices
    for (int x = 0; x < PROBLEM.states_number; x++){
        this->_set_rewards(x);
    }

    //-- verbose
    if (SEARCH.verbose >= utils::Verbose::high){
        cout << "=== reward shaping: " << endl;
        for (int x = 0; x < PROBLEM.states_number; x++){
            cout << "x " << x << endl;
            for (int joint_action = 0; joint_action < PROBLEM.actions_joint_number; joint_action++){
                double original_reward = PROBLEM.rewards_matrix(x, joint_action);
                cout << "\t ju " << joint_action
                    << " r(x,ju) " << original_reward;
                vector<int> actions = utils::get_indivIndices(PROBLEM.actions_number_byAgent, PROBLEM.agents_number, joint_action);
                Support supp_ju;
                double sum = 0;
                for (int agent: PROBLEM.agents){
                    supp_ju.set_iAction(agent, actions.at(agent));
                    double indiv_rew = this->rewards_matrices.at(agent)(x, supp_ju.get_jAction());
                    cout << " | agent " << agent
                         << " iu " << actions.at(agent)
                         << " r " << indiv_rew;
                    sum += indiv_rew;
                }
                cout << endl;
                assert(abs(original_reward - sum) < 1e-8);
            }
        }
    }
}


double Reward_Shaping::get_expected_reward(const Occupancy_State & oState, Decision_Rule & decision_rule) const{
    if (!SEARCH.use_reward_shaping)
        return oState.get_expected_reward(decision_rule);

    int agent = oState.get_next_agent();
    double reward = 0;
    const auto &rew_mat = this->rewards_matrices.at(agent);
    for (auto & support_proba: oState.get_supports_probas(oState.is_compressed)){
        auto supp = support_proba.first;
        int iu = oState.get_iAction(decision_rule, supp, agent); // individual action
        supp.set_iAction(agent, iu);
        int ju = supp.get_jAction(); // (partial) joint action : <u^1, u^2, ..., u^agent>
        reward += support_proba.second * rew_mat(supp.get_hiddenState(), ju);
    }
    return reward;
}


/// @brief Get the value stored in this->rewards_matrices.
/// @param agent Current agent number. 
/// @param x State number.
/// @param action The "partial joint action" corresponding to the agent and its predecessors,  i.e. u^{1:agent} .
/// @return The reward value.
double Reward_Shaping::get_reward(int agent, int x, int action) const{
    if (!SEARCH.use_reward_shaping){
        if (agent < PROBLEM.last_agent)
            return 0;
        else
            return PROBLEM.rewards_matrix(x, action);
    }
    
    return this->rewards_matrices.at(agent)(x, action);
}


/// @brief Recursively fill agent reward matrices:
/// the call with only the first argument, i.e. _set_reward(x),
/// fill r(x, <u_0, u_1, ...., u_i>) foreach i in [0, n-1].
void Reward_Shaping::_set_rewards(int x, int ju, int agent, double prev_sum){
    if (agent > PROBLEM.last_agent)
        return;
    
    Support supp_ju;
    supp_ju.set_jAction(ju);
    for (int iu = 0; iu < PROBLEM.actions_number_byAgent.at(agent); iu++){
        supp_ju.set_iAction(agent, iu);
        ju = supp_ju.get_jAction();
        double reward = this->_get_max_reward(x, ju, agent)
                        - prev_sum;
        this->rewards_matrices.at(agent)(x, ju) = reward; // fill the agent reward matrix
        this->_set_rewards(x, ju, agent + 1, prev_sum + reward); // recursion
    }
}

/// @brief Given the action of the first agents, compute the maximum reward for the state x,
/// i.e. compute: r(x, u^{:agent-1}) =  max_{u^{agent+1:}} r(x,<u^{:agent-1}, u^{agent+1:}>)
/// @param x a state
/// @param ju u^{:agent-1}
/// @param agent The last agent whose action is set.
/// @return r(x, u^{:agent-1})
double Reward_Shaping::_get_max_reward(int x, int ju, int agent) const{
    if (agent == PROBLEM.last_agent)
        return PROBLEM.rewards_matrix(x, ju);
    
    double max_reward = -INFINITY;
    Support supp_ju;
    supp_ju.set_jAction(ju);
    for (int iu = 0; iu < PROBLEM.actions_number_byAgent.at(agent + 1); iu++){
        supp_ju.set_iAction(agent + 1, iu);
        ju = supp_ju.get_jAction();
        double iu_value = this->_get_max_reward(x, ju, agent + 1);
        max_reward = max(max_reward, iu_value);
    }
    return max_reward;
}

}} // end namespace