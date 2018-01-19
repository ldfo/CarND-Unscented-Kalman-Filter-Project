#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	// define vector
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // check estimation size different from zero and it's size
    if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
        std::cout << "Invalid estimation or ground_truth!!\n";
        return rmse;
    }

    // accumulate residuals
    for (unsigned int i = 0; i < estimations.size(); ++i) {
        VectorXd residual = estimations[i] - ground_truth[i];
        // residual squared
        residual = residual.array() * residual.array();
        rmse += residual;
    }
    // calculate  mean
    rmse = rmse / estimations.size();
    // return sqrt of result
    return rmse.array().sqrt();
}