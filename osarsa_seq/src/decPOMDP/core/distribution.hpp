#pragma once

#include "../../core/_module.hpp"

namespace decPOMDP{
using namespace std;
using namespace algebra;

struct Distribution{

    Vector probabilities;

    Distribution();
    Distribution(size_t size_); // uniform distribution
    Distribution(const initializer_list<double> & distrib);
    Distribution(const Vector & distrib);

    double get_probability(int index) const;
    void set_probability(int index, double value);
    void normalize();
    int sample() const;
    int argmax() const;
    size_t size() const;

    friend ostream &operator<<(ostream &os, const Distribution &d){
        os << "[" << d.probabilities.transpose() << "]";
        return os;
    }
};

struct Distribution_Softmax: public Distribution{

    Vector activations;

    Distribution_Softmax(size_t size_);

    void set_probability(int index, double value); // rise an error (not allowed, see set_activations)
    
    //--  specific to Distribution_Softmax
    const Vector & get_activations() const;
    void set_activations(const Vector &);
    Vector get_logGradient(int index) const;
};


}