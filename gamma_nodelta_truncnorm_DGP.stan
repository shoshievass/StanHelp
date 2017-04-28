functions {
  matrix getExpectedUtilities(int N, int J, 
                              vector omega, vector inv_omega, 
                              vector gamma, 
                              matrix premiums, 
                              matrix deductibles) {
    matrix[J,N] expected_utilities_transpose;
    matrix[N, J] expected_utilities; // normalized wrt the first good
    matrix[N, J] dPi_acc;
    matrix[N, J] dPi_no_acc;
    matrix[N, J] utilities_acc;
    matrix[N, J] utilities_no_acc;
    
    //these should be N x J
    dPi_acc = rep_matrix(-1,N,J) .* (premiums + deductibles); 
    dPi_no_acc = rep_matrix(-1,N,J) .* premiums;
    
    utilities_acc = dPi_acc - rep_matrix(gamma*(0.5),J) .* dPi_acc .* dPi_acc;
    utilities_no_acc = dPi_no_acc - rep_matrix(gamma*(0.5), J) .* dPi_no_acc .* dPi_no_acc;

    
    expected_utilities = rep_matrix(omega, J) .* utilities_acc + rep_matrix(inv_omega, J) .* utilities_no_acc;

    expected_utilities_transpose = expected_utilities';
    return(expected_utilities_transpose);
  }
}

data {
  int N; // observations
  int M; // covariates
  int J; // plans
  matrix[N, M] X; // covariates
  // int plan_choice[N]; // plan choices ~categorical(softmax(V_ij))
  // int claim[N]; //claims outcomes as binomial(inv_logit(X*beta))
  matrix[N,J] premiums;
  matrix[N,J] deductibles;
  vector[N] delta;
}
parameters {
  // parameters
  vector[M] beta; // regression coefficients
  real beta_intercept; //intercept coefficient

  // hyper-parameters
  real<lower=0,upper=.5> mu_gamma; // mean of risk aversion
  // real z_mu_gamma;
  real<lower=0> sigma_gamma; // sd of risk aversion
  
  vector[N] z_gamma; 

}

transformed parameters {
  // Type parameters
  vector<lower=0>[N] gamma; // risk aversion parameter - partially pooled

  // Plan choice transformed parameters
  matrix<lower = 0,upper=1>[J, N] theta; // probability of each plan being chosen by i 

  vector<lower = 0, upper = 1>[N] mu; // probability of an accident given X_i
  vector[N] omega; // distorted probability of an accident given X_i
  vector[N] inv_omega; // 1 - omega

  matrix[J,N] exp_utilities; //expected utility from each plan (+ 0 for outside good)
  

  gamma = mu_gamma + sigma_gamma*z_gamma; // individual risk distortion

  mu = inv_logit(beta_intercept * rep_vector(1.0, N) + X*beta); 

  omega = delta .* mu;
  inv_omega = rep_vector(1,N) - omega;
  
  exp_utilities = getExpectedUtilities( N,  J,  omega, inv_omega, gamma, premiums, deductibles);
  {
    for(n in 1:N) {
      theta[1:J,n] = softmax((exp_utilities[1:J,n])); // Probability of choosing each plan
    }
  }
}

model {
  // priors
  beta ~ normal(0, 3); //accident prob coeffs
  beta_intercept ~ normal(0, 3); //accident prob coeffs
  
  mu_gamma ~ normal(0,1); // mean gamma prior
  sigma_gamma ~ normal(0, 2); // sd gamma prior

  to_vector(z_gamma) ~ normal(0, 1);

}

generated quantities {
  // Type parameters
  int plan_choice_sim[N];
  int claim_sim[N];
  
  for(n in 1:N) {
    plan_choice_sim[n] = categorical_rng(theta[1:J,n]); //plan choice likelihood
    claim_sim[n] = bernoulli_rng(mu[n]); // accident probability
  }
}
  
