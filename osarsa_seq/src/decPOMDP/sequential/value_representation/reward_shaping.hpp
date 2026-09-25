#pragma once

#include <vector>

#include "../../core/_module.hpp"
#include "../occupancy_state.hpp"


namespace decPOMDP{
namespace sequential{
    using namespace std;


    class Reward_Shaping{
        vector<Matrix<double>> rewards_matrices; // reward shaping by agent, in order to improve exploration
        int horizon_tau;

    public:
        Reward_Shaping();
        
        double get_expected_reward(const Occupancy_State &, Decision_Rule &) const;
        double get_reward(int agent, int x, int action) const;

    private:
        void _set_rewards(int x, int ju = 0, int agent = 0, double prev_sum = 0);
        double _get_max_reward(int x, int ju = 0, int agent = 0) const;
    };

}}
