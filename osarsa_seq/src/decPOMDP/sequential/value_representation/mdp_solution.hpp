#pragma once

#include "../occupancy_state.hpp"
#include "../../../MDP/Backward_induction.hpp"

#include <unordered_map>

namespace decPOMDP{
namespace sequential{
    using namespace std;

    class MDP_Solution{

        MDP::Backward_Induction mdp_solver;
        
    public:
        bool solved = false;
        
        MDP_Solution();

        void solve();

        double get_value(Occupancy_State &oState);
        double get_value(Support supp, int agent, int step) const;
        double get_value(const POMDP::Belief & belief, Support supp, int agent, int step) const;
        
    protected:
    };

}} // end namespace