#pragma once

#include "../../core/_module.hpp"
#include "../occupancy_state.hpp"


namespace decPOMDP{
namespace sequential{
using namespace std;


    class Qvalues_Tabular{
        Support_Val_Map container; // map: support -> value
        
        //-- for default values
        int step;
        int agent;

    public:
        Qvalues_Tabular();

        double get_value(const Support &) const;
        bool has_value(const Support &) const;
        void set_value(const Support &, double val);
        void clear();
        size_t size() const;
        
        friend ostream &operator<<(ostream &os, const Qvalues_Tabular & q_func);
        string get_repr(int step, int agent) const;
    };


    class Qvalues_Tabular_v2{
        Support_Val_Map container; // map: <x,o,u> -> value
        Support_Val_Map default_values; // map: <x,u> -> value
        map<Support, Support> labels;
        int agent;
        
    public:
        Qvalues_Tabular_v2(int agent);
        
        double get_value(const Support &) const;
        double get_default_value(const Support &) const;
        double get_default_value(int x, int ju) const;
        void set_value(const Support &, double val);
        void set_default_value(int x, int ju, double val);
        void set_labels(map<Support, Support> );
        Support get_label(Support) const;
        void clear();
        size_t size() const;


        string get_repr(int step, int agent) const;
        friend ostream &operator<<(ostream &os, const Qvalues_Tabular_v2 & q_func);
    };



}} // end namespace
