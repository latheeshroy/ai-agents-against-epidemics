#include "distribution.hpp"
#include <stdlib.h>     /* srand, rand */

namespace decPOMDP{
using namespace std;


Distribution::Distribution(){}

Distribution::Distribution(size_t size_){
    //-- uniform distribution
    double val = 1 / double(size_);
    this->probabilities  = Vector(size_, val);
}

Distribution::Distribution(const Vector & distrib): probabilities(distrib){}

Distribution::Distribution(const initializer_list<double> & distrib): probabilities(distrib){}

/// @brief To get a valid distribution (sum = 1)
void Distribution::normalize(){
    double sum = this->probabilities.sum();
    this->probabilities.array() /= sum;
}

/// @brief Return an index \in [0, size) 
/// following the distribution probabilities 
int Distribution::sample() const{
    double rnd = utils::fast_calc::rand();
    double sum = 0;
    int index = 0;

    int _size = this->probabilities.size();
    for (int i = 0; i < _size; i++){
        sum += this->probabilities[i];
        if (sum >= rnd)
            return index;
        index++;
    }
    return index - 1;
}

int Distribution::argmax() const{
    double max = -INFINITY;
    int arg = -1;

    int _size = this->probabilities.size();
    for (int i = 0; i < _size; i++){
        double val = this->probabilities[i];
        if (val > max){
            max = val;
            arg = i;
        }
    }
    return arg;
}

size_t Distribution::size() const{
    return this->probabilities.size();
}

double Distribution::get_probability(int index) const{
    return this->probabilities[index];
}
void Distribution::set_probability(int index, double value){
    this->probabilities[index] = value;
    this->normalize();
}


Distribution_Softmax::Distribution_Softmax(size_t size_): Distribution(size_){
    this->activations  = Vector(size_, 0);
}

void Distribution_Softmax::set_probability(int index, double value){
    cout << "Distribution_Softmax set_probability() not allowed ... " << index << " " << value << endl;
    throw -1;
}

const Vector & Distribution_Softmax::get_activations() const{
    return this->activations;
}

Vector Distribution_Softmax::get_logGradient(int index) const{
    int size_ = this->size();
    Vector gradient(size_);
    for (int i = 0; i < size_; i++){
        gradient[i] = utils::kronecker(i, index) - this->get_probability(index);
    }
    return gradient;
}


void Distribution_Softmax::set_activations(const Vector & values){
    //-- set activations
    this->activations = values;
    //-- update probabilities
    this->probabilities.array() = values.array().exp();
    this->normalize();
}


}