function plaid = BayesPlaid_cacheMCMC(plaid, mcmc_itr)
% plaid = plaid_cacheMCMC(plaid, mcmc_itr)
%
% Cache the MCMC histories. 
%
% input: 
% plaid       - BayesPlaid instance
% mcmc_itr    - iteration number of this Gibbs sweep
% 
% output: 
% plaid         - updated PLAID instance
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last Update: 17/11/15

%% load memories
  
% load cacheed variables
phi_gibbs = plaid.pphi_gibbs;
theta_gibbs = plaid.ttheta_gibbs;
lambda1_gibbs = plaid.llambda1_gibbs;
lambda2_gibbs = plaid.llambda2_gibbs;
z1_gibbs = plaid.zz1_gibbs;
z2_gibbs = plaid.zz2_gibbs;

% load current samples
phi = plaid.pphi;
theta = plaid.ttheta;
lambda1 = plaid.llambda1;
lambda2 = plaid.llambda2;
z1 = plaid.zz1;
z2 = plaid.zz2;

% Suf-stats??
% hyper-parameters??

%% get averaged
phi_gibbs = (mcmc_itr - 1) * phi_gibbs + phi;
theta_gibbs = (mcmc_itr - 1) * theta_gibbs + theta;
lambda1_gibbs = (mcmc_itr - 1) * lambda1_gibbs + lambda1;
lambda2_gibbs = (mcmc_itr - 1) * lambda2_gibbs + lambda2;
z1_gibbs = (mcmc_itr - 1) * z1_gibbs + z1;
z2_gibbs = (mcmc_itr - 1) * z2_gibbs + z2;

%% save
plaid.pphi_gibbs = phi_gibbs / mcmc_itr;
plaid.ttheta_gibbs = theta_gibbs / mcmc_itr;
plaid.llambda1_gibbs = lambda1_gibbs / mcmc_itr;
plaid.llambda2_gibbs = lambda2_gibbs / mcmc_itr;
plaid.zz1_gibbs = z1_gibbs / mcmc_itr;
plaid.zz2_gibbs = z2_gibbs / mcmc_itr;

