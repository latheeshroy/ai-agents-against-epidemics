#pragma once

#include "../../core/_module.hpp"

#include <iostream> 
#include <unordered_map>
#include <map>


namespace decPOMDP{
using namespace std;
using namespace common;


    /// @brief An occupancy state support is represented by a 64 bits number.
    class Support{
        uint64_t container = 0UL;

    // public:
        //-- bitset representation
        static int N_BITS_HIDDEN_STATE; // bits number for hidden state
        static int* N_BITS_iACTIONS; // bits number for a command by [agent]
        static int* N_BITS_iOBS; // bits number for an observation by [agent]
        
        static int** BITS_START_iOBS; // first bit of [step][agent] observations
        static int* BITS_START_iACTIONS; // first bit of U[agent]

        static uint64_t MASKS[64];
        static uint64_t** MASKS_iOBS; // MASKS_iOBS[step][agent]
        static uint64_t* MASKS_iACTIONS; // MASKS_iACTION[agent]
        static uint64_t MASK_ACTIONS;
        static uint64_t* MASKS_iHISTORIES;
        static uint64_t MASKS_jHISTORIES;
        static uint64_t MASK_HIDDEN_STATE; // hidden state = <x, last individuals u for intermediate agents>
        static uint64_t MASKS_jOBS; // the last observations (all individual last observation)

        static unordered_map<uint64_t, int> MAP_SUPPORTS_jACTIONS; // map: Support iACTIONS -> joint actions
        static uint64_t* jACTIONS;
        static uint64_t* jOBSERVATIONS;
        
    public:
        static void init();
        
        //-- constructors
        Support();
        Support(uint64_t);
        Support(int x, Support jh, int ju);

        //-- get methods
        uint64_t get_container() const; // return the container, i.e. the whole support
        int get_hiddenState() const;
        int get_observation(int agent, int oldness) const;
        int get_jAction()const;
        int get_iAction(int) const;
        vector<int> get_iActions() const;
        Support get_jHistory() const;
        Support get_partial(int first_agent) const;
        Support get_iHistory(int) const;
        Support get_iHist_masked(int) const;

        //-- set methods
        void set_hiddenState(int);
        void set_jObservation(int jObs);
        void set_iObservation(int agent, int observation);
        void set_iAction(int agent, int action);
        void set_jAction(int);
        void clear_actions();
        void set_iHistory(int, const Support&);
        void clear_iHistory(int);
        Support do_step(int hidden_state, int jObs) const;

    protected:
        void _shift_iObservations(int agent);

    public:
        //-- operators
        bool operator==(const Support &) const;
        bool operator!=(const Support &) const;
        bool operator<(const Support &) const;

        //-- print / debug    
        static void demo();
        friend ostream &operator<<(ostream &os, const Support &supp);
        string get_repr(int step = 100, int next_agent = 100) const;
        string get_ih_repr(int step, int agent) const;
        string get_zs_repr(int step, int agent) const;

    };

    using Support_Val_OrderMap = map<Support, double>;
    using Support_Val_Map = unordered_map<Support, double>;
    using Support_Int_Map = unordered_map<Support, int>;
    using Support_Set = unordered_set<Support>;

}// end namespace


//-- hash
namespace std {
    template <>
    struct hash<decPOMDP::Support> {
        size_t operator()(const decPOMDP::Support & supp) const{
            return supp.get_container();
        }
    };
}