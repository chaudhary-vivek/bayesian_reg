#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <tuple>


std::random_device rd;
std::mt19937 gen(rd());

double normal_pdf(double x, double mean, double stddev) {
    std::normal_distribution<> dist(mean, stddev);
    return std::exp(-0.5 * std::pow((x - mean) / stddev, 2)) / (stddev * std::sqrt(2.0 * M_PI));
}

double inv_gamma_pdf(double x, double shape, double scale) {
    if (x <= 0) return 0;
    double coef = std::pow(scale, shape) / std::tgamma(shape);
    return coef * std::pow(x, -shape - 1) * std::exp(-scale / x);
}

double prior_beta_0() {
    std::normal_distribution<> dist(0, 10);
    return dist(gen);
}

double prior_beta_1() {
    std::normal_distribution<> dist(0, 10);
    return dist(gen);
}

double prior_sigma2() {
    std::gamma_distribution<> dist(1, 1);  // Shape and rate
    return 1.0 / dist(gen);  // Convert gamma to inverse gamma
}

double likelihood(const std::vector<double>& X, const std::vector<double>& Y, double beta0, double beta1, double sigma2) {
    double sum_log_prob = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        double pred = beta0 + beta1 * X[i];
        double resid = Y[i] - pred;
        sum_log_prob += std::log(normal_pdf(resid, 0, std::sqrt(sigma2)));
    }
    return std::exp(sum_log_prob);
}

std::vector<std::tuple<double, double, double, double>> monte_carlo_posterior(const std::vector<double>& X, const std::vector<double>& Y, int num_samples) {
    std::vector<std::tuple<double, double, double, double>> samples;
    samples.reserve(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        double b0 = prior_beta_0();
        double b1 = prior_beta_1();
        double s2 = prior_sigma2();
        double post = likelihood(X, Y, b0, b1, s2) * normal_pdf(b0, 0, 10) * normal_pdf(b1, 0, 10) * inv_gamma_pdf(s2, 1, 1);
        samples.emplace_back(b0, b1, s2, post);
    }
    return samples;
}

std::tuple<std::vector<double>, std::vector<double>> generate_data(int n_samples, double beta_0, double beta_1, double sigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(0.0, 10.0);  // Uniform distribution for X between 0 and 10
    std::normal_distribution<> dis_eps(0.0, sigma);     // Normal distribution for epsilon with mean 0 and std sigma

    std::vector<double> X(n_samples);
    std::vector<double> Y(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        X[i] = dis_x(gen);                          // Generate data for X
        double epsilon = dis_eps(gen);              // Generate noise epsilon
        Y[i] = beta_0 + beta_1 * X[i] + epsilon;    // Calculate Y based on the linear model
    }

    return std::make_tuple(X, Y);
}

extern std::vector<std::tuple<double, double, double, double>> monte_carlo_posterior(const std::vector<double>& X, const std::vector<double>& Y, int num_samples);

int main() {
    double beta_0 = 2.0;
    double beta_1 = 3.0;
    int n_samples = 100;
    double sigma = 1.0;  // Standard deviation of the noise

    // Generate data
    auto [X, Y] = generate_data(n_samples, beta_0, beta_1, sigma);

    // Number of samples to generate in the Monte Carlo simulation
    int num_samples = 10000;

    // Perform the Monte Carlo posterior sampling
    auto samples = monte_carlo_posterior(X, Y, num_samples);
    
    // Output the first few samples to verify correctness
    std::cout << "Displaying the first 10 samples from the posterior distribution:" << std::endl;
    for (int i = 0; i < 10 && i < samples.size(); ++i) {
        auto& sample = samples[i];
        std::cout << "Sample " << i + 1 << ": "
                  << "Beta0 = " << std::get<0>(sample) << ", "
                  << "Beta1 = " << std::get<1>(sample) << ", "
                  << "Sigma2 = " << std::get<2>(sample) << ", "
                  << "Posterior = " << std::get<3>(sample) << std::endl;
    }

    return 0;
}