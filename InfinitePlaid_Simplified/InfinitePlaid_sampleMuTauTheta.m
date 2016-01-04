function plaid = InfinitePlaid_sampleMuTauTheta(plaid)
% plaid = InfinitePlaid_sampleMuTauTheta(plaid)
%
% Sample Mu and Tau, Normal dist. parameters for submatrices from
% flat Normal and Gamma prior assumption.
%
% Importance sampling with re-parameterization.
%
% input, output: plaid - MATLAB intance of InfinitePlaid model.
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last Update: 17/11/15 (dd/mm/yy)
EPS = 0.000001;
DEBUG = 0;

%% load variables

K = plaid.KK;

theta = plaid.ttheta;

mu_theta = plaid.mmu_theta;
tau_theta = plaid.ttau_theta;

%% sampling

% new memory
new_mu_theta = sampleMu(mu_theta, tau_theta, K, theta);
new_tau_theta = sampleTau(mu_theta, tau_theta, K, theta);
if( DEBUG )
    if(mu_theta ~= new_mu_theta)
        display(['InfinitePlaid_sampleMuTauTheta: mu_theta changed ', num2str(mu_theta), ...
            ' -> ', num2str(new_mu_theta)]);
    end
    
    if(tau_theta ~= new_tau_theta)
        display(['InfinitePlaid_sampleMuTauTheta: tau_theta changed ', num2str(tau_theta), ...
            ' -> ', num2str(new_tau_theta)]);
    end
end

%% save
plaid.mmu_theta = new_mu_theta;
plaid.ttau_theta = new_tau_theta;

% end of sampleAB

%% internal functions
function new_mu = sampleMu(mu, tau, K, theta)

% pre-compute averages and variances
bar_theta = mean(theta);
S = var(theta, 1) * K;

% Normal hyper prior
normal_mu = 0.0;
normal_tau = 0.001;
num_sample = 10;

% re-parameterize for uniform sampling
inv_values = 0.00001 + 0.99 * rand(1, num_sample);
signs = zeros(1, num_sample);
for ii=1:num_sample
    signs(ii) = (-1)^ii;
end
normal_values = [ signs.*(inv_values ./ (1 - inv_values)) mu ];

log_prob_normal_values = zeros(size(normal_values));
for ii=1:length(normal_values)
    
    % prior
    log_prob_normal_values(ii) = ...
        -0.5 * normal_tau * (normal_values(ii) - normal_mu) * (normal_values(ii) - normal_mu);
    
    % likelihood
    log_prob_normal_values(ii) = log_prob_normal_values(ii) ...
        -0.5 * tau * ( K * (bar_theta - normal_values(ii)) * (bar_theta - normal_values(ii)) + S);
    
end % end ii-for
log_prob_normal_values = log_prob_normal_values - max(log_prob_normal_values);

% sample
pp = exp(log_prob_normal_values);
pp = pp / sum(pp);

rr = rand;
ss = 1 + sum(rr > cumsum(pp));
if(ss > num_sample + 1)
    ss = num_sample + 1;
end

new_mu = normal_values(ss);
% do not allow large moves
if (new_mu - mu) > tau * 3
    new_mu = mu + tau * 3;    
elseif (mu - new_mu) < tau * 3
    new_mu = mu - tau * 3;
end

%% sample Tau
function new_tau = sampleTau(mu, tau, K, theta)

% pre-compute averages and variances
bar_theta = mean(theta);
S = var(theta, 1) * K;

% Gamma hyper prior
gamma_a = 1.1;
gamma_lambda = 0.33; % larger expectation = larger variance
num_sample = 10;

% re-parameterize for uniform sampling
inv_values = 0.00001 + 0.99 * rand(1, num_sample);
gamma_values = [ (inv_values ./ (1 - inv_values)) tau ];

bar_theta = bar_theta - mu;

log_prob_gamma_values = zeros(size(gamma_values));
for ii=1:length(gamma_values)
    
    % prior
    log_prob_gamma_values(ii) = ...
        (gamma_a - 1) * log(gamma_values(ii)) ...
        - gamma_lambda * gamma_values(ii);
    
    % likelihood
    log_prob_gamma_values(ii) = log_prob_gamma_values(ii) ...
        + 0.5 * K * log(gamma_values(ii)) ...
        -0.5 * gamma_values(ii) * ( K * bar_theta * bar_theta + S);
    
end
log_prob_gamma_values = log_prob_gamma_values - max(log_prob_gamma_values);

% sample
pp = exp(log_prob_gamma_values);
pp = pp / sum(pp);

rr = rand;
ss = 1 + sum(rr > cumsum(pp));
if(ss > num_sample + 1)
    ss = num_sample + 1;
end

new_tau = gamma_values(ss);
% do not allow large moves
if new_tau > tau * 2
    new_tau = tau * 2;    
elseif new_tau < tau * 0.5
    new_tau = tau * 0.5;
end
