#include "common.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <random>


namespace common{
using namespace std;
using namespace algebra;

double Problem_Data::get_y_proba(int x, int j_action, int y){
    const auto & y_probas = this->dynamics_T[j_action][x];
    auto got = y_probas.find(y);
    return got == y_probas.end() ? 0 : got->second;
}

double Problem_Data::get_z_proba(int y, int j_action, int z){
    const auto & z_probas = this->dynamics_O[j_action][y];
    auto got = z_probas.find(z);
    return got == z_probas.end() ? 0 : got->second;
}

double Problem_Data::get_proba_u_x_z(int u, int x, int z){
    const auto & z_probas = this->probas_u_x_z[u][x];
    auto got = z_probas.find(z);
    return got == z_probas.end() ? 0 : got->second;
}

const vector<Problem_Data::Reachable> & Problem_Data::get_reachables(int x, int j_action){
    if (!this->reachable_generator)
        return this->reachables_from_x_u.at(x).at(j_action);
    {
        lock_guard<mutex> lock(this->lazy_reachables_mutex);
        const auto &cached = this->lazy_reachables.at(x).at(j_action);
        if (cached)
            return *cached;
    }
    auto generated = make_shared<const vector<Reachable>>(
        this->reachable_generator(x, j_action));
    lock_guard<mutex> lock(this->lazy_reachables_mutex);
    auto &cached = this->lazy_reachables.at(x).at(j_action);
    if (!cached)
        cached = generated;
    return *cached;
}


void Problem_Data::finalize(bool set_obs_joint2indiv){
    this->reachable_generator = {};
    this->lazy_reachables.clear();
    this->agents.clear();
    for (int agent = 0; agent < this->agents_number; agent++)
        this->agents.push_back(agent);

    if (set_obs_joint2indiv){
        this->_set_obs_joint2indiv();
    }
    this->_finalize_probas();

    if (SEARCH.verbose >= utils::Verbose::medium){
        cout << "Problem: "
            << " states " << this->states_number
            << " actions " << this->actions_joint_number
            << " obs " << this->observations_joint_number
            << endl; 
    }
}

void Problem_Data::finalize_lazy(){
    this->reachable_generator = {};
    this->lazy_reachables.clear();
    this->agents.clear();
    for (int agent = 0; agent < this->agents_number; agent++)
        this->agents.push_back(agent);
    this->_set_obs_joint2indiv();
    this->lazy_reachables.assign(
        this->states_number,
        vector<shared_ptr<const vector<Reachable>>>(this->actions_joint_number));
}

void Problem_Data::_finalize_probas(){
    this->reachables_from_x.resize(this->states_number);
    for (int x = 0; x < this->states_number; x++){
        vector<vector<Reachable>> vec_x;
        auto & reach_from_x = this->reachables_from_x.at(x);
        for (int u = 0; u < this->actions_joint_number; u++){
            vector<Reachable> vec_u;
            double weight=0, weight2=0;
            for (auto const & y_proba: this->dynamics_T[u][x]){
                int y = y_proba.first;
                double proba_y = y_proba.second;
                weight += proba_y;
                for (auto const & z_proba: this->dynamics_O[u][y]){
                    double proba = proba_y * z_proba.second;
                    weight2 += proba;
                    int z = z_proba.first;
                    if (this->get_proba_u_x_z(u, x, z) == 0)
                        this->probas_u_x_z[u][x][z] = proba;
                    else
                        this->probas_u_x_z[u][x][z] += proba;
                    auto indiv_obs = this->observations_joint2indiv[z];
                    vec_u.push_back(Reachable{y, proba, z});
                    reach_from_x.insert(make_pair(y, z));
                }
            }
            vec_x.push_back(vec_u);

            //-- check accuracy
	        assert(abs(weight - 1) < 1e-5);
            assert(abs(weight2 - 1) < 1e-5);
        }
        reachables_from_x_u.push_back(vec_x);
    }

}

void Problem_Data::_set_obs_joint2indiv_recur(int agent, vector<int> &obs_sequence){
    if (agent == this->agents_number){
        int obs_joint = utils::get_jointIndex(obs_sequence, this->observations_number_byAgent, this->agents_number);
        this->observations_joint2indiv[obs_joint] = obs_sequence;
        return;
    }
    for (int obs = 0; obs < this->observations_number_byAgent[agent]; obs++){
        vector<int> new_sequence = obs_sequence;
        new_sequence.push_back(obs);
        _set_obs_joint2indiv_recur(agent + 1, new_sequence);
    }
}

/// @brief build observations_joint2indiv to find individual observations from a joint number
void Problem_Data::_set_obs_joint2indiv(){
    this->observations_joint2indiv.resize(this->observations_joint_number);
    vector<int> empty_seq;
    this->_set_obs_joint2indiv_recur(0, empty_seq);
}



string get_problem_repr(){
    using namespace utils;
    auto color = Colors::magenta;
    ostringstream msg;

    //-- problem title
    if (SEARCH.verbose){
        msg << _get_colorStr({color, Colors::bold})
            << "PROBLEM: " << PROBLEM.bench_name
            << " horizon " << SEARCH.horizon 
            << " agents " << PROBLEM.agents_number
            << _get_colorStr(Colors::reset)
            << endl;
    }

    //-- bench infos
    if (SEARCH.verbose >= Verbose::medium){
        msg << _get_colorStr(color)
            << "\t states_number: " << PROBLEM.states_number
            << " actions_joint_number: " << PROBLEM.actions_joint_number
            << " observations_joint_number: " << PROBLEM.observations_joint_number
            << endl
            << "\t belief_init: " << PROBLEM.belief_init.transpose()
            << _get_colorStr(Colors::reset)
            << endl;
    }

    return msg.str();
}

string get_search_repr(){
    using namespace utils;
    auto color = Colors::cyan;
    ostringstream msg;
    if (SEARCH.verbose){
        msg << _get_colorStr({color, Colors::bold})
            << "ALGO: " << SEARCH.algo_name
            << " LPE " << (SEARCH.compress ? "yes": "no")
            << " MemoryLength " << SEARCH.truncation
            << " Workers " << SEARCH.workers
            << " LazyReachability " << (SEARCH.lazy_reachability ? "yes" : "no")
            << _get_colorStr(Colors::reset)
            << endl;
    }

    return msg.str();
}


}