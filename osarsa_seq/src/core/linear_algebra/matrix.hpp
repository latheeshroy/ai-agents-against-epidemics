#pragma once

#include <eigen3/Eigen/Dense>


namespace algebra{
  template <typename T = double>
  using dynMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template <typename T = double>
  class Matrix: public dynMatrix<T>{
    public:
    Matrix(): dynMatrix<T>(0, 0){};
    Matrix(size_t rows, size_t cols): dynMatrix<T>(rows, cols){};
    Matrix(size_t rows, size_t cols, T val): dynMatrix<T>(rows, cols){
      this->init(val);
    };

    void init(T val){
      this->setConstant(val);
    }

    T max() const{
      return this->maxCoeff();
    }

    T min() const{
      return this->minCoeff();
    }

    /// @brief return the max value of a the i^th col (resp. the i^th row) for dim = 0 (resp. dim = 1)  
    T max(int dim, int i){
      return dim == 0 ? this->col(i).maxCoeff(): this->row(i).maxCoeff();
    }

    /// @brief return the min value of a the i^th col (resp. the i^th row) for dim = 0 (resp. dim = 1)  
    T min(int dim, int i){
      return dim == 0 ? this->col(i).minCoeff(): this->row(i).minCoeff();
    }
  };
}
