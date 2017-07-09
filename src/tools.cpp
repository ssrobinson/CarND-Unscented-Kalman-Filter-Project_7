#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // Check the measurement inputs
  if (estimations.size() == 0 || 
      estimations.size() != ground_truth.size()) {
    cout << "Error - estimation or ground_truth data invalid" << endl;
    return rmse;
  }

  // Accumulate squared residuals
  for (int i = 0; i < estimations.size(); ++i){

    VectorXd residual = estimations[i] - ground_truth[i];

    // Calculate residuals
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // Calculate the mean RMSE
  rmse = rmse / estimations.size();

  // Calculate the squared root of mean RMSE
  rmse = rmse.array().sqrt();

  return rmse;
}
