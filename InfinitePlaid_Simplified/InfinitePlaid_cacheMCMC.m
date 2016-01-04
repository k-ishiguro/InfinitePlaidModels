function plaid = InfinitePlaid_cacheMCMC(plaid, mcmc_itr)
% plaid = InfinitePlaid_cacheMCMC(plaid, mcmc_itr)
%
% Cache the MCMC histories.
%
% input:
% plaid       - InfinitePlaid instance
% mcmc_itr    - iteration number of this Gibbs sweep
%
% output:
% plaid         - updated PLAID instance
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last Update: 17/11/15 (dd/mm/yy)

%% load memories

% load cacheed variables
K_gibbs = plaid.KK_gibbs;
phi_gibbs = plaid.pphi_gibbs;
theta_gibbs = plaid.ttheta_gibbs;
z1_gibbs = plaid.zz1_gibbs;
z2_gibbs = plaid.zz2_gibbs;

% load current samples
K = plaid.KK;
phi = plaid.pphi;
theta = plaid.ttheta;
z1 = plaid.zz1;
z2 = plaid.zz2;

%% get averaged
phi_gibbs = (mcmc_itr - 1) * phi_gibbs + phi;

if K < K_gibbs
    
    theta_gibbs(1:K) = (mcmc_itr - 1) * theta_gibbs(1:K) + theta;
    z1_gibbs(:, 1:K) = (mcmc_itr - 1) * z1_gibbs(:, 1:K) + z1;
    z2_gibbs(:, 1:K) = (mcmc_itr - 1) * z2_gibbs(:, 1:K) + z2;
    
    theta_gibbs(K+1:K_gibbs) = (mcmc_itr - 1) * theta_gibbs(K+1:K_gibbs);
    z1_gibbs(:, K+1:K_gibbs) = (mcmc_itr - 1) * z1_gibbs(:, K+1:K_gibbs);
    z2_gibbs(:, K+1:K_gibbs) = (mcmc_itr - 1) * z2_gibbs(:, K+1:K_gibbs);
    
else
    
    theta_gibbs2 = (mcmc_itr - 1) * theta_gibbs + theta(1:K_gibbs);
    z1_gibbs2 = (mcmc_itr - 1) * z1_gibbs + z1(:, 1:K_gibbs);
    z2_gibbs2 = (mcmc_itr - 1) * z2_gibbs + z2(:, 1:K_gibbs);
    
    theta_gibbs = theta;
    z1_gibbs = z1;
    z2_gibbs = z2;
    theta_gibbs(1:K_gibbs) = theta_gibbs2;
    z1_gibbs(:, 1:K_gibbs) = z1_gibbs2;
    z2_gibbs(:, 1:K_gibbs) = z2_gibbs2;
    
    K_gibbs = K;
    
end

%% save
plaid.pphi_gibbs = phi_gibbs / mcmc_itr;
plaid.ttheta_gibbs = theta_gibbs / mcmc_itr;
plaid.zz1_gibbs = z1_gibbs / mcmc_itr;
plaid.zz2_gibbs = z2_gibbs / mcmc_itr;
plaid.KK_gibbs = K_gibbs;

