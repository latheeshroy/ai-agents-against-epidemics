#pragma once

#include "../core/_module.hpp"


namespace MDP{
    using namespace std;
    using namespace algebra;

    using State_Set = unordered_set<int>;


    class Backward_Induction{
        vector<Vector> state_values; // for each time-step
        vector<vector<Vector>> state_Qvalues; // for each time-step, foreach joint-action

    public:
        Backward_Induction();

        void solve();
        double get_value(int time_step, int state) const;
        double get_value(int time_step, const Vector & belief) const;
        double get_Qvalue(int time_step, int action, int state) const;
        double get_Qvalue(int time_step, int action, const Vector & belief) const;

        static vector<State_Set> get_reachable_states(); // for each time-step

    protected:
        void _set_value(int time_step, int state, double value);
        void _set_Qvalue(int time_step, int state, int action, double value);

    };

}