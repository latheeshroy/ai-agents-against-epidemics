#pragma once

#include "../core/_module.hpp"

namespace POMDP{
using namespace std;
using namespace common;

struct Belief: Vector{

    Belief();
    Belief(size_t size_, double val);
    Belief(const initializer_list<double> &);
    Belief(const Vector &);
    Belief(const Belief &, int action);

    unordered_map<int, Belief> get_reacheable_beliefs(int action) const;
    double getExpectedReward(int action) const;
    double normalize();
    
    bool operator==(const Vector & other) const;

    friend ostream &operator<<(ostream &os, const Belief &b){
        os << b.transpose();
        return os;
    }
};

/*struct Belief_Hash{
    size_t operator()(Belief const& b) const {      
        size_t seed = 0; 
        for (auto val: b){
            utils::hash_combine(seed, int(val * 1e7));
        }
        return seed;
    }
};

struct Belief_Hash_Approx{
    size_t operator()(Belief const& b) const {      
        size_t seed = 0;
        for (auto val: b){
            utils::hash_combine(seed, int(val * 1e4));
        }
        return seed;
    }
};*/

struct Belief_Hash{
    static Vector weights;
    static void init();
    size_t operator()(Belief const& belief) const;
};


}