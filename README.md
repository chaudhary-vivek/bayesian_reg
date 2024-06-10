## Experiment setup

 

The following experment is set up in a [Jupyter notebook](https://github.com/chaudhary-vivek/bayesian_reg/blob/main/from_scratch.ipynb), and a [C++](https://github.com/chaudhary-vivek/bayesian_reg/blob/main/test2.cpp) script.

 

## Synthetic data

 

We genrate random values for x and generate y values using the following equation:

 

<p style="text-align: center;"> y = $\beta$<sub>0</sub> + $\beta$<sub>1</sub>x + ε </p>

 

Where:

 

$\beta$<sub>0</sub> = 2

 

$\beta$<sub>1</sub> = 3

 

Standard deviation of noise ε = 1

 

## Priors

 

The priors for $\beta$<sub>0</sub> are assumed to be normally distributed with mean 0, and standard deviation 10.

 

The priors for $\beta$<sub>1</sub> are assumed to be normally distributed with mean 0, and standard deviation 10.

 

The priors for ε are assumed to have inverse gaussian distribution with standard deviation 1.

 

## Bayesian Inference

 

10000 samples are drawn from the posterior using MCMC

 

## Results

 

The estimated values of:

 

$\beta$<sub>0</sub> = 2.166892014785945 (Actual is 2)

 

$\beta$<sub>1</sub> = 2.9566554039577913 (Actual is 3)

 

Standard deviation of ε = 0.8220087340441655 (Actual is 1)

 

## C++ is AT LEAST 60X faster than Python
