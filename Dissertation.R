###########################
#####data simulation#######
###########################

#######sales simulation######
for (t in 1:end_time) {
  for (i in 1:6) {
    lambda_i <- mu[i]
    if (t > 1) {
      lambda_i <- lambda_i + sum(sapply(1:(t-1), function(k) ifelse(product_events[k, i] > 0, alpha_self * exp(-beta_self * (t - k)), 0)))
      for (j in setdiff(1:6, i)) {
        lambda_i <- lambda_i + sum(sapply(1:(t-1), function(k) ifelse(product_events[k, j] > 0, alpha_cross * exp(-beta_cross * (t - k)), 0)))
      }
    }
    
    product_events[t, i] <- if (runif(1) < lambda_i) {
      rnbinom(1, size, prob) 
    } else {
      0
    }
  }
}


#####price simulation######
for (t in 2:days) {
  for (product in 1:num_products) {
    if (runif(1) < prob_change) {
      change <- rnorm(1, mean = mean_change, sd = sd_change)
      prices[t, product] <- prices[t - 1, product] + change
    } else {
      prices[t, product] <- prices[t - 1, product]
    }
  }
}

###########################
#######zero process########
###########################

##########Benchmark########
zero_day_probabilities <- sapply(zero_process[-1], function(x) mean(x == 0))
predicted_probabilities_Bench <- matrix(0, nrow = nrow(test_data[-1]), ncol = ncol(test_data[-1]))
for (i in 1:6) {
  predicted_probabilities_Bench[, i] <- zero_day_probabilities[i]
}
actual_sales <- test_zero_process[-1]  
pl_z_Bench <- numeric(ncol(actual_sales))
for (i in 1:ncol(actual_sales)) {
  log_likelihood_sum <- 0
  for (t in 1:nrow(actual_sales)) {
    p_its <- predicted_probabilities_Bench[t, i]
    E_it <- actual_sales[t, i]
    log_likelihood_t <- log(p_its ^ E_it * (1 - p_its) ^ (1 - E_it))
    log_likelihood_sum <- log_likelihood_sum + log_likelihood_t
  }
  pl_z_Bench[i] <- log_likelihood_sum / nrow(actual_sales)
}


###### Baseline model #######

stan_code <- "
data {
  int<lower=0> N;  
  int<lower=0> K; 
  int<lower=0, upper=1> y[N, K];  
}

parameters {
  vector[K] theta;  
}

model {
  theta ~ normal(0, 1);  
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(theta);
  }
}
"

####### HBz model #######
stan_code <- "
data {
  int<lower=0> N;
  int<lower=0> K;  
  matrix[N, K] log_price;  
  matrix[N, 11] s_kt;  
  int<lower=0, upper=1> y[N, K];  
}

parameters {
  real rho_1;  
  vector[12] rho_rest;  
  matrix[K, 13] theta;  
}

model {
  rho_1 ~ normal(0, 1);
  rho_rest ~ normal(0, 1);
  
  vector[13] rho;
  rho[1] = rho_1;
  for (i in 2:13)
    rho[i] = rho_rest[i-1];
  
  for (i in 1:13) {
    theta[, i] ~ normal(rho[i], sqrt(0.5));
  }

  for (n in 1:N) {
    for (k in 1:K) {
      real phi_i_t = theta[k, 1] + theta[k, 2] * log_price[n, k] + dot_product(theta[k, 3:13], s_kt[n]);
      y[n, k] ~ bernoulli_logit(phi_i_t);
    }
  }
}
"

####### BEz model #######
stan_code <- "
data {
  int<lower=0> N; 
  int<lower=0> K;  
  matrix[N, K] log_price;  
  matrix[N, 11] s_kt;  
  int<lower=0, upper=1> y[N, K];  
  matrix[N, K] history; 
}

parameters {
  vector[K] theta_1; 
  matrix[K, 12] theta_rest;  
  vector<lower=0>[K] kappa;  
  vector<lower=0>[K] mu_raw; 
  vector<lower=0>[K] tau;  
}

transformed parameters {
  matrix[K, 13] theta;
  vector<lower=1>[K] mu;
  for (k in 1:K) {
    theta[k, 1] = theta_1[k];
    for (j in 2:13) {
      theta[k, j] = theta_rest[k, j - 1];
    }
    mu[k] = 1 + mu_raw[k];
  }
}

model {
  theta_1 ~ normal(0, 1);
  to_vector(theta_rest) ~ normal(0, 1);
  
  kappa ~ gamma(3, 1);
  mu_raw ~ gamma(5, 1);
  tau ~ gamma(10, 5);

  for (n in 1:N) {
    for (k in 1:K) {
      real S_it = 0.0;
      for (s in 1:(n-1)) {
        if (s > 0 && s <= N && k > 0 && k <= K && history[s, k] > 0) {
          real dt = n - s;
          if (dt > 1) {
            real g = (tgamma(dt-2+tau[k]) / (tgamma(dt-1) * tgamma(tau[k]))) *
                     pow((mu[k] - 1) / (mu[k] + tau[k]), dt-1) *
                     pow(tau[k] / (mu[k] + tau[k]), tau[k]);
            S_it += kappa[k] * history[s, k] * g;
          }
        }
      }
      real phi_i_t = theta[k, 1] + theta[k, 2] * log_price[n, k] + dot_product(theta[k, 3:13], s_kt[n]) + S_it;
      y[n, k] ~ bernoulli_logit(phi_i_t);
    }
  }
}

"

##### HBEz model #####
stan_code <- "
data {
  int<lower=0> N;  
  int<lower=0> K; 
  matrix[N, K] log_price;  
  matrix[N, 11] s_kt;  
  int<lower=0, upper=1> y[N, K];  
  matrix[N, K] history;  
}

parameters {
  vector[13] rho;  
  matrix[K, 13] theta;  
  real<lower=0> eta_1;  
  vector<lower=0>[K] eta_2_raw;  
  real<lower=0> eta_3;  

  vector<lower=0>[K] kappa;  
  vector<lower=0>[K] mu_raw;  
  vector<lower=0>[K] tau;  
}

transformed parameters {
  vector<lower=1>[K] mu;  
  vector<lower=0>[K] eta_2;  

  for (k in 1:K) {
    mu[k] = 1 + mu_raw[k];  
    eta_2[k] = 1 + eta_2_raw[k];  
  }
}

model {
  rho[1] ~ normal(0, 1);  
  for (j in 2:13) {
    rho[j] ~ normal(0, 1);  
  }

  for (k in 1:K) {
    for (j in 1:13) {
      theta[k, j] ~ normal(rho[j], sqrt(0.5));  
    }
    kappa[k] ~ gamma(eta_1, 1);  
    mu_raw[k] ~ gamma(eta_2, 1);  
    tau[k] ~ gamma(eta_3, 5);  
  }

  eta_1 ~ gamma(50, 10);  
  eta_2_raw ~ gamma(10, 10);  
  eta_3 ~ gamma(500, 50);  

  for (n in 1:N) {
    for (k in 1:K) {
      real S_it = 0.0;
      for (s in 1:(n-1)) {
        if (history[s, k] > 0) {
          real dt = n - s;
          if (dt > 1) {
            real g = (tgamma(dt-2+tau[k]) / (tgamma(dt-1) * tgamma(tau[k]))) *
                     pow(mu[k] - 1 / (mu[k] + tau[k]), dt-1) *
                     pow(tau[k] / (mu[k] + tau[k]), tau[k]);
            S_it += kappa[k] * history[s, k] * g;
          }
        }
      }
      real phi_i_t = theta[k, 1] + theta[k, 2] * log_price[n, k] + dot_product(theta[k, 3:13], s_kt[n]) + S_it;
      y[n, k] ~ bernoulli_logit(phi_i_t);
    }
  }
}

"

####### BECz model #######
stan_code <- "
data {
  int<lower=0> N;  
  int<lower=0> K;  
  matrix[N, K] log_price;  
  matrix[N, 11] s_kt;  
  int<lower=0, upper=1> y[N, K];  
  matrix[N, K] history;  
  matrix[N, K] cross_history;  
}

parameters {
  vector[K] theta_1;
  matrix[K, 12] theta_rest;
  vector<lower=0>[K] kappa;
  vector<lower=0>[K] mu_raw;
  vector<lower=0>[K] tau;

  vector<lower=0>[K] cross_kappa;
  vector<lower=0>[K] cross_mu_raw;
  vector<lower=0>[K] cross_tau;
}

transformed parameters {
  matrix[K, 13] theta;
  vector<lower=1>[K] mu;
  vector<lower=1>[K] cross_mu;
  for (k in 1:K) {
    theta[k, 1] = theta_1[k];
    for (j in 2:13) {
      theta[k, j] = theta_rest[k, j - 1];
    }
    mu[k] = 1 + mu_raw[k];
    cross_mu[k] = 1 + cross_mu_raw[k];
  }
}

model {
  theta_1 ~ normal(0, 1);
  to_vector(theta_rest) ~ normal(0, 1);
  
  kappa ~ gamma(3, 1);
  mu_raw ~ gamma(5, 1);
  tau ~ gamma(10, 5);
  
  cross_kappa ~ gamma(5, 10);
  cross_mu_raw ~ gamma(5, 1);
  cross_tau ~ gamma(10, 5);

  for (n in 1:N) {
    for (k in 1:K) {
      real S_it = 0.0;
      real cross_S_it = 0.0;

      for (s in 1:(n-1)) {
        if (history[s, k] > 0) {
          real dt = n - s;
          if (dt > 1) {
            real g = (tgamma(dt-2+tau[k]) / (tgamma(dt-1) * tgamma(tau[k]))) *
                     pow((mu[k] - 1) / (mu[k] + tau[k]), dt-1) *
                     pow(tau[k] / (mu[k] + tau[k]), tau[k]);
            S_it += kappa[k] * history[s, k] * g;
          }
        }
        if (cross_history[s, k] > 0) {
          real dt_cross = n - s;
          if (dt_cross > 1) {
            real g_cross = (tgamma(dt_cross-2+cross_tau[k]) / (tgamma(dt_cross-1) * tgamma(cross_tau[k])))*
                     pow((cross_mu[k] - 1) / (cross_mu[k] + cross_tau[k]), dt_cross - 1) *
                     pow(cross_tau[k] / (cross_mu[k] + cross_tau[k]), cross_tau[k]);
            cross_S_it += cross_kappa[k] * cross_history[s, k] * g_cross;
          }
        }
      }
      
      real phi_i_t = theta[k, 1] + theta[k, 2] * log_price[n, k] + dot_product(theta[k, 3:13], s_kt[n]) + S_it + cross_S_it;
      y[n, k] ~ bernoulli_logit(phi_i_t);
    }
  }
}

"

##### HBECz model ########
stan_code <- "
data {
  int<lower=0> N;  
  int<lower=0> K;  
  matrix[N, K] log_price;  
  matrix[N, 11] s_kt;  
  int<lower=0, upper=1> y[N, K]; 
  matrix[N, K] history;  
  matrix[N, K] cross_history;  
}

parameters {
  vector[13] rho;
  matrix[K, 13] theta;
  real<lower=0> eta_1;
  vector<lower=0>[K] kappa;
  real<lower=0> eta_2;
  vector<lower=0>[K] mu_raw;
  real<lower=0> eta_3;
  vector<lower=0>[K] tau;

  real<lower=0> eta_1_prime;
  vector<lower=0>[K] cross_kappa;
  real<lower=0> eta_2_prime;
  vector<lower=0>[K] cross_mu_raw;
  real<lower=0> eta_3_prime;
  vector<lower=0>[K] cross_tau;
}

transformed parameters {
  vector<lower=1>[K] mu;
  vector<lower=1>[K] cross_mu;

  for (k in 1:K) {
    mu[k] = 1 + mu_raw[k];
    cross_mu[k] = 1 + cross_mu_raw[k];
  }
}

model {
  rho[1] ~ normal(0,1);
  for (j in 2:13) {
    rho[j] ~ normal(0, 1);
  }

  for (k in 1:K) {
    theta[k] ~ normal(rho, sqrt(0.5)); 
  }

  eta_1 ~ gamma(50, 10);
  kappa ~ gamma(eta_1, 1);
  eta_2 ~ gamma(10, 10);
  mu_raw ~ gamma(eta_2, 1);
  eta_3 ~ gamma(500, 50);
  tau ~ gamma(eta_3, 5);

  eta_1_prime ~ gamma(40, 20);
  cross_kappa ~ gamma(eta_1_prime, 10);
  eta_2_prime ~ gamma(10, 10);
  cross_mu_raw ~ gamma(eta_2_prime, 1);
  eta_3_prime ~ gamma(500, 50);
  cross_tau ~ gamma(eta_3_prime, 5);

  for (n in 1:N) {
    for (k in 1:K) {
      real S_it = 0.0;
      real cross_S_it = 0.0;

      for (s in 1:(n-1)) {
        if (history[s, k] > 0) {
          real dt = n - s;
          if (dt > 1) {
            real g = (tgamma(dt-2+tau[k]) / (tgamma(dt-1) * tgamma(tau[k]))) *
                     pow((mu[k] - 1) / (mu[k] + tau[k]), dt-1) *
                     pow(tau[k] / (mu[k] + tau[k]), tau[k]);
            S_it += kappa[k] * history[s, k] * g;
          }
        }
        if (cross_history[s, k] > 0) {
          real dt_cross = n - s;
          if (dt_cross > 1) {
            real g_cross = (tgamma(dt_cross-2+cross_tau[k]) / (tgamma(dt_cross-1) * tgamma(cross_tau[k])))*
                     pow((cross_mu[k] - 1) / (cross_mu[k] + cross_tau[k]), dt_cross - 1) *
                     pow(cross_tau[k] / (cross_mu[k] + cross_tau[k]), cross_tau[k]);
            cross_S_it += cross_kappa[k] * cross_history[s, k] * g_cross;
          }
        }
      }
      
      real phi_i_t = theta[k, 1] + theta[k, 2] * log_price[n, k] + dot_product(theta[k, 3:13], s_kt[n]) + S_it + cross_S_it;
      y[n, k] ~ bernoulli_logit(phi_i_t);
    }
  }
}
"


###########################
#######count process#######
###########################

######## Benchmark ######## 
train_distribution <- train_long_data %>%
  filter(sales > 0) %>%  # 
  group_by(product, sales) %>%
  summarize(count = n()) %>%
  ungroup() %>%
  group_by(product) %>%
  mutate(empirical_prob = count / sum(count))  

plc_i <- test_long_data %>%
  filter(sales > 0) %>%  
  left_join(train_distribution, by = c("product", "sales")) %>% 
  mutate(empirical_prob = ifelse(is.na(empirical_prob), 0.5 / sum(train_long_data$sales > 0), empirical_prob)) %>% 
  summarize(total_log_likelihood = sum(log(empirical_prob), na.rm = TRUE))


###### Baseline model #######
stan_code <- "
data {
  int<lower=0> N;  
  int<lower=0> no_models;  
  int<lower=0> y_count[N, no_models];  
}

parameters {
  real phi_c[no_models];  
}

model {
  phi_c ~ normal(0, 1); 
  
  for (m in 1:no_models) {
    for (n in 1:N) {
      real mu = exp(phi_c[m]);  
      y_count[n, m] ~ neg_binomial_2(mu, 1);  
    }
  }
}
"

###### HBc model ######
stan_code <- "
data {
  int<lower=0> N;  
  int<lower=0> K;  
  matrix[N, K] log_price;  
  int<lower=0> y[N, K];  
}

parameters {
  vector[K] theta1; 
  vector[K] theta2;  
  real rho1;  
  real rho2;  
}

transformed parameters {
  matrix[N, K] phi;  

  for (i in 1:K) {
    for (t in 1:N) {
      phi[t, i] = exp(theta1[i] + theta2[i] * log_price[t, i]); 
    }
  }
}

model {
  rho1 ~ normal(0, 1);
  rho2 ~ normal(0, 1);

  for (i in 1:K) {
    theta1[i] ~ normal(rho1, sqrt(0.75));
    theta2[i] ~ normal(rho2, sqrt(0.75));
    for (t in 1:N) {
      y[t, i] ~ neg_binomial_2(phi[t, i], 1);  
    }
  }
}
"

###### BEc model #####
stan_code <- "
functions {
  real g_function(int t, real mu, real tau) {
    real binom_coeff = exp(binomial_coefficient_log(t - 2 + tau, t - 1));  
    real term1 = pow((mu - 1) / (mu - 1 + tau), t - 1);
    real term2 = pow(tau / (mu - 1 + tau), tau);
    return binom_coeff * term1 * term2;
  }
}

data {
  int<lower=0> N;  
  int<lower=0> K;  
  matrix[N, K] log_price;  
  int<lower=0> y[N, K];  
  matrix[N, K] history;  
}

parameters {
  vector[K] theta_1;  
  vector[K] theta_2; 
  vector<lower=0>[K] kappa;  
  vector<lower=0>[K] mu_raw; 
  vector<lower=0>[K] tau;  
}

transformed parameters {
  matrix[N, K] phi;  
  matrix[N, K] S;  
  vector[K] mu;  
  matrix[N, K] lambda;  

  for (k in 1:K) {
    mu[k] = 1 + mu_raw[k];  
    for (n in 1:N) {
      phi[n, k] = theta_1[k] + theta_2[k] * log_price[n, k];
      S[n, k] = 0;

      for (s in 1:(n-1)) {
        S[n, k] += kappa[k] * history[s, k] * g_function(n - s, mu[k], tau[k]);
      }
    }
  }
}

model {
  theta_1 ~ normal(1, sqrt(0.75));
  theta_2 ~ normal(-1, sqrt(0.75));
  kappa ~ gamma(1, 5);
  mu_raw ~ gamma(3, 1);
  tau ~ gamma(4, 1);

  for (k in 1:K) {
    for (n in 1:N) {
      y[n, k] ~ neg_binomial_2(exp(phi[n, k] + S[n, k]), 1);
    }
  }
}
"

###### HBEc model #####
stan_code <- "
functions {
  real g_function(int t, real mu, real tau) {
    real binom_coeff = exp(binomial_coefficient_log(t - 2 + tau, t - 1));  
    real term1 = pow((mu - 1) / (mu - 1 + tau), t - 1);
    real term2 = pow(tau / (mu - 1 + tau), tau);
    return binom_coeff * term1 * term2;
  }
}

data {
  int<lower=0> N;  
  int<lower=0> K;  
  matrix[N, K] log_price;  
  int<lower=0> y[N, K]; 
  matrix[N, K] history;  
}

parameters {
  real rho1;  
  real rho2;  
  real<lower=0> eta1;  
  real<lower=0> eta2;  
  real<lower=0> eta3;  

  vector[K] theta_1;  
  vector[K] theta_2;  
  vector<lower=0>[K] kappa;  
  vector<lower=0>[K] mu_raw;  
  vector<lower=0>[K] tau;  
}


transformed parameters {
  matrix[N, K] phi; 
  matrix[N, K] S;  
  vector[K] mu;  
  for (k in 1:K) {
    mu[k] = 1 + mu_raw[k];  
    for (n in 1:N) {
      phi[n, k] = theta_1[k] + theta_2[k] * log_price[n, k];
      S[n, k] = 0; 

      for (s in 1:(n-1)) {
        S[n, k] += kappa[k] * history[s, k] * g_function(n - s, mu[k], tau[k]);
      }
    }
  }
}
model {

  rho1 ~ normal(0, 1);
  rho2 ~ normal(0, 1);
  eta1 ~ gamma(50, 10);
  eta2 ~ gamma(10, 10);
  eta3 ~ gamma(500, 50);

  theta_1 ~ normal(rho1, sqrt(0.75));
  theta_2 ~ normal(rho2, sqrt(0.75));
  kappa ~ gamma(eta1, 5);
  mu_raw ~ gamma(eta2, 1);
  tau ~ gamma(eta3, 1);

  for (k in 1:K) {
    for (n in 1:N) {
      y[n, k] ~ neg_binomial_2(exp(phi[n, k] + S[n, k]), 1.0); 
    }
  }
}
"























