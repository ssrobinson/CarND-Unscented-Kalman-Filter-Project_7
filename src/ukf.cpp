#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_ << 1, 1, 1, 1, 0.1;

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ <<   0.2, 0, 0, 0, 0,
	  0, 0.2, 0, 0, 0,
	  0, 0, 1, 0, 0,
	  0, 0, 0, 1, 0,
	  0, 0, 0, 0, 1;

  /// State dimension
  n_x_ = 5;

  /// Augmented state dimension
  n_aug_ = 7;

  /// predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  /// time when the state is true, in us
  time_us_ = 0.0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.9;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /// Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);

  /// Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  /// the current NIS for radar
  NIS_radar_ = 0.0;

  /// the current NIS for laser
  NIS_laser_ = 0.0;
  
  /**
  
  Complete the initialization. See ukf.h for other member properties.

  */
}

UKF::~UKF() {}



/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
	// Initialize laser, radar, and timestamp measurements
	if (!is_initialized_) {
		
		time_us_ = meas_package.timestamp_;

		if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {

			x_(0) = meas_package.raw_measurements_(0);
			x_(1) = meas_package.raw_measurements_(1);
		}

		else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
			
			float rho = meas_package.raw_measurements_(0);
			float phi = meas_package.raw_measurements_(1);
			float rho_dot = meas_package.raw_measurements_(2);
			x_(0) = rho * cos(phi);
			x_(1) = rho * sin(phi);
		}

		// Finised with initialization
		is_initialized_ = true;

		return;

	}
	// calculate elapsed time from previous measurement to current measurement
	float dt = (meas_package.timestamp_ - time_us_) / 1000000.0; // dt converted to seconds
	time_us_ = meas_package.timestamp_;

	Prediction(dt);

	// Update Radar and Laser measurements

	if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		UpdateLidar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		UpdateRadar(meas_package);
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
	
	/// Generate Sigma Points ///
	
	// Create sigma point matrix
	MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

	// Calculate square root of P matrix
	MatrixXd A = P_.llt().matrixL();

	// Set lambda for non-augmented sigma points
	lambda_ = 3 - n_x_;

	// Set first column of sigma point matrix
	Xsig.col(0) = x_;

	// Set remaining sigma points
	for (int i = 0; i < n_x_; i++) {
		Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
		Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
	}

	
	///  Generate Augmented Sigma Points ///
	
	// Create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.fill(0); // Initialize vector with zeros

	// Create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.fill(0); // Initialize matrix with zeros

	// Create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	// Set lambda for augmented sigma points
	lambda_ = 3 - n_aug_;

	// Create augmented mean state
	x_aug.head(n_x_) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	// Create augmented covariance matrix
	P_aug.fill(0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(5, 5) = std_a_ * std_a_;
	P_aug(6, 6) = std_yawdd_ * std_yawdd_;

	// Calculate square root of P matrix
	MatrixXd L = P_aug.llt().matrixL();

	// Create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	double sqrt_lambda_n_aug = sqrt(lambda_ + n_aug_); 
	VectorXd sqrt_lambda_n_aug_L;

	for (int i = 0; i< n_aug_; i++)
	{
		sqrt_lambda_n_aug_L = sqrt_lambda_n_aug * L.col(i);
		Xsig_aug.col(i + 1) = x_aug + sqrt_lambda_n_aug_L;
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt_lambda_n_aug_L;
	}

	
	/// Predict Sigma Points ///
	
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		// Extract values
		double p_x = Xsig_aug(0, i);
		double p_y = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);

		// Predicted state values
		double px_p, py_p;

		// Avoid division by zero
		double v_yawd = v / yawd;
		double term = yaw + yawd * delta_t;
		double v_delta_t = v * delta_t;
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v_yawd * (sin(term) - sin(yaw));
			py_p = p_y + v_yawd * (cos(yaw) - cos(term));
		}
		else {
			px_p = p_x + v_delta_t * cos(yaw);
			py_p = p_y + v_delta_t * sin(yaw);
		}

		double v_p = v;
		double yaw_p = term;
		double yawd_p = yawd;
		// Include noise
		px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
		py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
		v_p = v_p + nu_a * delta_t;
		yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
		yawd_p = yawd_p + nu_yawdd * delta_t;

		// Write predicted sigma point into right column

		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}

	
	/// Convert Predicted Sigma Points to state mean vector and state covariance matrix
	
	// Set weights
	double weight_0 = lambda_ / (lambda_ + n_aug_);
	weights_(0) = weight_0;
	for (int i = 1; i < 2 * n_aug_ + 1; i++) {  
		double weight = 0.5 / (n_aug_ + lambda_);
		weights_(i) = weight;
	}

	// Predicted state mean vector
	x_.fill(0);            
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // iterate over all augmented sigma points
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	// Predicted state covariance matrix
	P_.fill(0);       
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // iterate over all augmented sigma points

		// Calculate difference of state vector x_
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		// Normalize angles between -pi and pi
		while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;
		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
	// Extract measurement as z vector 
	VectorXd z = meas_package.raw_measurements_;

	// Set lidar measurement dimension to "2" for p_x and p_y
	int n_z = 2;

	// Create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	// Transform sigma points into measurement space for 2n+1 sigma points
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  

		// Extract values 
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);

		// Lidar measurement model
		Zsig(0, i) = p_x;
		Zsig(1, i) = p_y;
	}

	// Mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	// Measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  

		// Residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	// Add noise to measurement covariance matrix
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_laspx_*std_laspx_, 0,
		0, std_laspy_*std_laspy_;
	S = S + R;

	// Create x_corr matrix for cross correlation
	MatrixXd x_corr = MatrixXd(n_x_, n_z);

	
	/// UKF Update for Lidar ///
	

	// Calculate cross correlation matrix
	x_corr.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
		// Residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// Calculate difference of state vector x
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		x_corr = x_corr + weights_(i) * x_diff * z_diff.transpose();
	}

	// Kalman gain K;
	MatrixXd K = x_corr * S.inverse();

	// Residual
	VectorXd z_diff = z - z_pred;

	// Calculate NIS
	NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

	// Update state mean vector x_ and covariance matrix P_
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
	// Extract measurement as z vector
	VectorXd z = meas_package.raw_measurements_;

	// Set measurement dimension to "3" for radar (rho, phi, and rho_dot)
	int n_z = 3;

	// Create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	// Transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  

		// Extract values 
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);
		double v1 = cos(yaw) * v;
		double v2 = sin(yaw) * v;

		// Radar measurement model
		Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);  // rho                        
		Zsig(1, i) = atan2(p_y, p_x);          //phi                              
		Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   //rho_dot
	}

	// Mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	// Measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
		// Residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// Normalize angles between -pi and pi
		while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	// Add noise to measurement covariance matrix S
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_radr_*std_radr_, 0, 0,
		0, std_radphi_*std_radphi_, 0,
		0, 0, std_radrd_*std_radrd_;
	S = S + R;

	// Create x_corr matrix for cross correlation 
	MatrixXd x_corr = MatrixXd(n_x_, n_z);

	
	///  UKF Update for Radar ///
	
	// Calculate cross correlation matrix
	x_corr.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  

		// Residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		// Normalize angles between -pi and pi
		while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

		// Calculate difference of state vector x
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		// Normalize angles between -pi and pi
		while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

		x_corr = x_corr + weights_(i) * x_diff * z_diff.transpose();
	}

	// Kalman gain K;
	MatrixXd K = x_corr * S.inverse();

	// Residual
	VectorXd z_diff = z - z_pred;

	// Normalize angles between -pi and pi
	while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
	while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

	// Calculate NIS
	NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

	// Update state mean vector x_ and covariance matrix P_
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();
}
