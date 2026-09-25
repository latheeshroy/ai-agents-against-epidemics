#include "Q_representation.hpp"


namespace decPOMDP{
namespace sequential{
using namespace std;


Qvalues_Tabular::Qvalues_Tabular(){}


/// @brief The size of the container, i.e. the number of Qvalues stored in the mapping.
size_t Qvalues_Tabular::size() const{
    return this->container.size();
}


/// @brief Look for the q value associated to a tuple (x,o,u^,iu). \\
/// @brief where iu is the individual action and u^ denotes u^{:agent-1} (sequential framework). \\
/// @brief If this value is not found, return 0 (safe initialization fallback).
/// @param supp a concise representation of <x,o,u^,iu>.
double Qvalues_Tabular::get_value(const Support & supp) const{
    const auto got = this->container.find(supp);
    if (got == this->container.end())
        return 0.0;
    return got->second;
}


bool Qvalues_Tabular::has_value(const Support & supp) const{
    return this->container.find(supp) != this->container.end();
}


void Qvalues_Tabular::clear(){
    this->container.clear();
}


/// @param supp = concise representation of <x,o,u^,iu>.
void Qvalues_Tabular::set_value(const Support & supp, double val){
    this->container[supp] = val;
}


ostream &operator<<(ostream &os, const Qvalues_Tabular & q_func){
    for (auto & supp_val: q_func.container){
        os  << supp_val.first
            << " val: " <<  supp_val.second << endl;
    }
    return os;
}

string Qvalues_Tabular::get_repr(int step, int agent) const{
    ostringstream os;
    for (auto & supp_val: this->container){
        os  << supp_val.first.get_repr(step, agent + 1) 
            << " val: " <<  supp_val.second << endl;
    }
    return os.str();
}


///////////////////////////////////////
// v2
///////////////////////////////////////
Qvalues_Tabular_v2::Qvalues_Tabular_v2(int agent): agent(agent){}


/// @brief The size of the container, i.e. the number of Qvalues stored in the mapping.
size_t Qvalues_Tabular_v2::size() const{
    return this->container.size();
}


/// @brief Look for the q value associated to a tuple (x,o,u^,iu). \\
/// @brief where iu is the individual action and u^ denotes u^{:agent-1} (sequential framework). \\
/// @brief If this value is not found, return a default value.
/// @param supp a concise representation of <x,o,u^,iu>.
double Qvalues_Tabular_v2::get_value(const Support & supp) const{
    Support label = this->get_label(supp);
    const auto got = this->container.find(label);
    if (got != this->container.end())
        return got->second;
    return this->get_default_value(label);
}


double Qvalues_Tabular_v2::get_default_value(const Support & supp) const{
    return this->get_default_value(supp.get_hiddenState(), supp.get_jAction());
}


double Qvalues_Tabular_v2::get_default_value(int x, int ju) const{
    Support xu(x, 0, ju);
    return this->default_values.at(xu);
}


/// @brief Clear the mapping: <x,o,u> -> value  \\
/// @brief filled with set_value() \\
/// @brief AND the pointer on the corresponding oState \\
/// @brief BUT NOT the default values filled with set_default_value()
void Qvalues_Tabular_v2::clear(){
    this->container.clear();
    this->labels.clear();
}


void Qvalues_Tabular_v2::set_labels(map<Support, Support> labels){
    this->labels = labels;
}


Support Qvalues_Tabular_v2::get_label(Support supp) const{
    int iu = supp.get_iAction(this->agent);
    Support supp_ = supp; supp_.set_iAction(this->agent, 0);

    auto got = this->labels.find(supp_);
    if (got == this->labels.end())
        return supp;
    Support label = got->second;
    label.set_iAction(this->agent, iu);
    return label;
}


void Qvalues_Tabular_v2::set_value(const Support & supp, double val){
    this->container[supp] = val;
}


void Qvalues_Tabular_v2::set_default_value(int x, int ju, double val){
    Support xu(x, 0, ju);
    this->default_values[xu] = val;
}


ostream &operator<<(ostream &os, const Qvalues_Tabular_v2 & q_func){
    for (auto & supp_val: q_func.container){
        os << supp_val.first << "  val: " <<  supp_val.second << endl;
    }
    return os;
}

string Qvalues_Tabular_v2::get_repr(int step, int agent) const{
    ostringstream os;
    os << " container_size " << this->container.size() << " default_size " << this->default_values.size() << endl;
    for (auto & supp_val: this->container){
        os << supp_val.first.get_repr(step, agent + 1)
            << "  val: " << supp_val.second
            << endl;
    }
    for (auto & supp_val: this->default_values){
        os << supp_val.first.get_repr(0, agent + 1) // don't print history (default_values depend on x and u)
            << "  val: " << supp_val.second
            << " (default value)"
            << endl;
    }
    return os.str();
}


}} // end namespace
