#pragma once

#include <eigen3/Eigen/Dense>



namespace algebra{  
  using namespace std;
  
    //-- colVector is an Eigen::Matrix of dimension: size x 1
  template <typename T = double>
  using colVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  //-- _Vector iherit form colVector, so it is an Eigen::Matrix.  
  //--    Hence one can also use inherited methods/operators from Eigen:
  //--    e.g. resize(int), size(), sum(), ==, []
  template <typename T = double>
  struct _Vector: public colVector<T>{

    _Vector(): colVector<T>(0){};

    _Vector(size_t size): colVector<T>(size){};

    _Vector(size_t size, T val): colVector<T>(size){
      this->setConstant(val);
    };

    _Vector(const initializer_list<T> & list): colVector<T>(list.size()){      
      int i = 0;
      for (const T val: list){
        (*this)(i++) = val;
      }
    }

    _Vector(const vector<T> & list): colVector<T>(list.size()){      
      int i = 0;
      for (const T val: list){
        (*this)(i++) = val;
      }
    }
    
    T max() const{
      return this->maxCoeff();
    }

    T min() const{
      return this->minCoeff();
    }

    T operator^(const _Vector& other) const {
      return this->dot(other);
    }
    
    T operator<=(const _Vector& other) const {
      return ((this->array() - other.array()) <= 0).all();
    }

    T operator+(const _Vector& other) const{
      return this->array() + other.array();
    }
    
    void expand(int n_val){
      this->conservativeResize(this->size() + n_val);  
    }
    
  };

  using Vector = _Vector<double>;
  using Vector_int = _Vector<int>;

  
  
}
