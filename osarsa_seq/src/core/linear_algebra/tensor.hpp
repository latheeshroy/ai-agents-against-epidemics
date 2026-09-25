#pragma once
#include <eigen3/unsupported/Eigen/CXX11/Tensor>


namespace algebra{
  using dynTensor1 = Eigen::Tensor<double, 1>;
  using dynTensor2 = Eigen::Tensor<double, 2>;
  using dynTensor3 = Eigen::Tensor<double, 3>;
  using dynTensor4 = Eigen::Tensor<double, 4>;
  
  class Tensor1: public dynTensor1{
    public:
    Tensor1(): dynTensor1(0){};
    Tensor1(int dim1): dynTensor1(dim1){};
    Tensor1(int dim1, double val): Tensor1(dim1){
        this->setConstant(val);
    }
  };

  class Tensor3: public dynTensor3{
    public:
    Tensor3(): dynTensor3(0, 0, 0){};
    Tensor3(int dim1, int dim2, int dim3): dynTensor3(dim1, dim2, dim3){};
    Tensor3(int dim1, int dim2, int dim3, double val): Tensor3(dim1, dim2, dim3){
        this->setConstant(val);
    }
  };

  class Tensor4: public dynTensor4{
    public:
    Tensor4(): dynTensor4(0, 0, 0, 0){};
    Tensor4(int dim1, int dim2, int dim3, int dim4): dynTensor4(dim1, dim2, dim3, dim4){};
    Tensor4(int dim1, int dim2, int dim3, int dim4, double val): Tensor4(dim1, dim2, dim3, dim4){
        this->setConstant(val);
    }
  };
}
