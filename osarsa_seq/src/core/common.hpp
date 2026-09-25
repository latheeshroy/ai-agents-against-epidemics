#pragma once

#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <sstream>


#include "utils.hpp"
#include "linear_algebra/Vector.hpp"
#include "linear_algebra/matrix.hpp"


namespace common{
using namespace std;
using namespace algebra;

    /// @brief Problem definition.
    /// Contains data: discount factor, agents name, reward matrix ...
    /// x is the current state , y the next state, z the observation, u the control value
    struct Problem_Data{

        //-- general (names are optional)
        double discount;
        bool criterion_reward;
        int agents_number;
        vector<int> agents;
        vector<string> agents_name; // optional
        string bench_name;
        //-- states
        int states_number = -1;
        Vector belief_init;
        vector<string> states_name; // optional
        unordered_map<string, int> names2states; // optional
        //-- observations
        int observations_joint_number;
        vector<int> observations_number_byAgent;
        vector<vector<string>> observations_names_byAgent; // optional
        vector<unordered_map<string, int>> agents_names2obs; // optional
        //-- actions
        int actions_joint_number;  
        vector<int> actions_number_byAgent;   
        vector<vector<string>> actions_names_byAgent; // optional
        vector<unordered_map<string, int>> agents_names2actions; // optional
        //-- rewards  
        Matrix<double> rewards_matrix; // (x,u)
        //-- dynamics
        unordered_map<int, unordered_map<int, unordered_map<int, double>>> dynamics_T; // dynamics_T[u][x][y]: p(y |u, x)
        unordered_map<int, unordered_map<int, unordered_map<int, double>>> dynamics_O; // dynamics_O[u][y][z]: p(z |u,y)
        
        //-- additional data to allow fast computation (set in finalize())
        int last_agent;
        unordered_map<int, unordered_map<int, unordered_map<int, double>>> probas_u_x_z; // p(z |x,u)
        vector<vector<int>> observations_joint2indiv;
        struct Reachable{
            int y;
            double proba;
            int joint_obs;
        };
        vector<vector<vector<Reachable>>> reachables_from_x_u; // reachables_from_x_u[x][u] = a vector of Reachable
        vector<unordered_set<pair<int, int>, utils::Pair_Hash<int,int>>> reachables_from_x; // reachables_from_x[x] = a set of (x, obs)
        function<vector<Reachable>(int, int)> reachable_generator;
        vector<vector<shared_ptr<const vector<Reachable>>>> lazy_reachables;
        mutex lazy_reachables_mutex;

        //-- utils
        double get_y_proba(int x, int j_action, int y);
        double get_z_proba(int y, int j_action, int z);
        double get_proba_u_x_z(int u, int x, int z);
        const vector<Reachable> & get_reachables(int x, int j_action);

        //-- to compute additional data
        void finalize(bool set_obs_joint2indiv = true);
        void finalize_lazy();

    protected:
        void _finalize_probas();
        void _set_obs_joint2indiv();
        void _set_obs_joint2indiv_recur(int agent, vector<int> &obs_sequence);

    }inline PROBLEM;


    struct Search_Parameters{
        //-- general
        int seed;
        int horizon;
        double timeout; // in seconds
        string log_filename;
        utils::Verbose verbose;
        string algo_name;
        //-- compression
        bool compress;
        int truncation;
        double compress_threshold;
        //-- RL
        float epsilon_start;
        //-- oSarsa
        bool use_reward_shaping; // for sequential-oSarsa
        bool use_portfolio;
        bool use_simulatedAnnealing;
        int workers = 1; // total workers, including the central learner
        bool lazy_reachability = false;
        //-- optional multiple initial beliefs
        string beliefs_file;

        void start(){
            utils::seed_init(this->seed, this->verbose);
            this->reset_time();
        }

        void reset_time(){
            MARKTIME
        }

        double get_time(){
            return TIME;
        }

        bool is_timeout(){
            return TIME > this->timeout;
        }


    }inline SEARCH;


    //-- print
    string get_problem_repr();
    string get_search_repr();
}
