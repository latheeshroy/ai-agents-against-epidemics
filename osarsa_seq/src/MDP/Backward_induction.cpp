#include "Backward_induction.hpp"

namespace MDP{
    using namespace std;
    using namespace algebra;
    using namespace common;


Backward_Induction::Backward_Induction():
    state_values(SEARCH.horizon + 1, Vector(PROBLEM.states_number, 0)),
    state_Qvalues(SEARCH.horizon + 1, vector(PROBLEM.actions_joint_number, Vector(PROBLEM.states_number, 0))){
}

void Backward_Induction::solve(){
    if (SEARCH.verbose)
        cout << utils::get_solver_repr("MDP:Backward_Induction:solve()") << endl;

    SEARCH.start();

    auto state_sets = Backward_Induction::get_reachable_states();
    //-- verbose
    if (SEARCH.verbose >= utils::Verbose::medium){
        cout << "reachable states number (by step): ";
        for (int step = 0; step <= SEARCH.horizon; step++)
            cout << state_sets.at(step).size() << " ";
        cout << endl;
    }

    //-- backup
    for (int step = SEARCH.horizon - 1; step >= 0 ; step--){
        for (const int & x: state_sets.at(step)){
            double best_val = -INFINITY;  
            for (int u = 0; u < PROBLEM.actions_joint_number; u++){
                double val = PROBLEM.rewards_matrix(x, u);
                for (const auto &y_proba: PROBLEM.dynamics_T[u][x]){
                    val += PROBLEM.discount * y_proba.second
                        * this->get_value(step + 1, y_proba.first);
                }
                this->_set_Qvalue(step, x, u, val);
                best_val = max(best_val, val);
            }
            this->_set_value(step, x, best_val);
        }
    }
    double elapsed_time = SEARCH.get_time();

    if (SEARCH.verbose){
        cout << "Initial belief value: " << this->get_value(0, PROBLEM.belief_init) << endl;
        cout << "Time elapsed: " << elapsed_time << endl;
    }
}

/// @brief Fill state_sets (reachable states per time-step from initial belief).
vector<State_Set> Backward_Induction::get_reachable_states(){
    vector<State_Set> state_sets(SEARCH.horizon + 1);

    //-- initial
    for (int x = 0; x < PROBLEM.states_number; x++){
        if (PROBLEM.belief_init[x] > 0)
            state_sets.at(0).insert(x);
    }

    //-- reachables
    for (int step = 0; step < SEARCH.horizon; step++){
        State_Set & state_set_next = state_sets.at(step + 1);
        for (int x: state_sets.at(step)){
            for (int u = 0; u < PROBLEM.actions_joint_number; u++){
                for (const auto &y_proba: PROBLEM.dynamics_T[u][x])
                    state_set_next.insert(y_proba.first);
            }
        }    
    }

    return state_sets;
}


double Backward_Induction::get_value(int time_step, int state)const{
    return this->state_values.at(time_step)[state];
}


double Backward_Induction::get_value(int time_step, const Vector & belief) const{
    return this->state_values.at(time_step).dot(belief);
}


double Backward_Induction::get_Qvalue(int time_step, int action, int state) const{
    return this->state_Qvalues.at(time_step).at(action)[state];
}


double Backward_Induction::get_Qvalue(int time_step, int action, const Vector & belief) const{
    return this->state_Qvalues.at(time_step).at(action).dot(belief);
}

void Backward_Induction::_set_value(int time_step, int state, double value){
    this->state_values.at(time_step)[state] = value;
}


void Backward_Induction::_set_Qvalue(int time_step, int state, int action, double value){
    this->state_Qvalues.at(time_step).at(action)[state] = value;
}


}