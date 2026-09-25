#include "belief.hpp"

namespace POMDP{
using namespace std;
using namespace common;

Belief::Belief(){}

Belief::Belief(size_t size_, double val): Vector(size_, val){}

Belief::Belief(const Vector & belief): Vector(belief){}

Belief::Belief(const initializer_list<double> & beliefs): Vector(beliefs){}

Belief::Belief(const Belief & old_belief, int action): Belief(old_belief.size(), 0){
    assert(PROBLEM.states_number == (int)old_belief.size());
    for (int x = 0; x < PROBLEM.states_number; x++){
        double x_proba = old_belief[x];
        for(int y = 0; y < PROBLEM.states_number; y++){
            this->operator[](y) += x_proba * PROBLEM.get_y_proba(x, action, y);
        }
    }
}

/// @brief Normalize to obtain a vector whose sum is 1.
/// @return The sum before normalization.
double Belief::normalize(){
    double sum = this->sum();
    this->array() /= sum;
    return sum;
}

/// @brief Compute the new beliefs reachables from (belief, action)
/// @return A map: obs -> new_belief
/// @warning THE BELIEFS ARE NOT NORMALIZED 
unordered_map<int, Belief> Belief::get_reacheable_beliefs(int action) const{
    unordered_map<int, Belief> new_beliefs; // map: obs -> new_belief

    Belief belief_null(this->size(), 0);

    for (int x = 0; x < PROBLEM.states_number; x++){
        if (this->operator[](x) < 1e-8)
            continue;
        for (const auto & reached: PROBLEM.get_reachables(x, action)){
            double proba = reached.proba * this->operator[](x);
            auto got = new_beliefs.find(reached.joint_obs);
            if (got != new_beliefs.end()){
                got->second[reached.y] += proba;
            }
            else{
                Belief new_belief = belief_null;
                new_belief[reached.y] = proba;
                new_beliefs.emplace(reached.joint_obs, new_belief);
            }
        }
    }

    return new_beliefs;
}


double Belief::getExpectedReward(int action) const{
    return PROBLEM.rewards_matrix.col(action).dot(*this);
}


bool Belief::operator==(const Vector & other) const{
    static const double approx_factor = 1e6;
    assert(this->size() == other.size());
    for (int i = 0; i < (int)this->size(); i++){
        if (int(this->operator[](i) * approx_factor) != int(other[i] * approx_factor))
            return false;
    }
    return true;
}


////////////////////////
// hash
////////////////////////

Vector Belief_Hash::weights;

void Belief_Hash::init(){
    weights = Vector(PROBLEM.states_number);
    int weight = 1;
    for (auto & val: weights)
        val = weight++;
}

size_t Belief_Hash::operator()(Belief const& belief) const {
    return size_t(belief.dot(weights) * 1e9);
}


}