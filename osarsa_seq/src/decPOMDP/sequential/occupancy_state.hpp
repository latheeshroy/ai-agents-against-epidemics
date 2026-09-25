#pragma once

#include "../core/_module.hpp"

#include <iostream>

namespace decPOMDP{
namespace sequential{
using namespace std;

    using jHist_BeliefActions_Map = unordered_map<Support, pair<POMDP::Belief, vector<int>>>;

    class Occupancy_State: public Occupancy_State_Base{

    public:
        Occupancy_State();
        Occupancy_State(const Vector & belief_init); // for root occupancy state
        Occupancy_State(int step, int next_agent); // for successors (next oState)
        virtual void test(){}
        ~Occupancy_State(){}
        
        Occupancy_State do_step(Decision_Rule &)const; // one agent is playing
        vector<Occupancy_State> do_step(vector<Decision_Rule> &)const; // all agent are playing
        double get_expected_reward(Decision_Rule & ) const;
        static int get_tau(int step, int agent);
        static pair<int, int> get_step_agent(int tau);
        int get_tau() const;

        unordered_map<Support, Support_Val_Map> get_transitions() const;
        bool allReachable(const unordered_map<Support, Support_Val_Map> & transitions, bool compact=true) const;

        friend ostream &operator<<(ostream &os, const Occupancy_State &s);
        string get_repr_short() const;

    protected:
    };

    using Trajectory = vector<Occupancy_State>;

    class Occupancy_State_Corner: public Occupancy_State{

    public:
        Occupancy_State_Corner(int step, int next_agent, const Support & support);
        Occupancy_State_Corner(const Occupancy_State &, const Support &);
    };


}} // end namespace
