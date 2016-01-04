function plaid = BayesPlaid_sampleTau0(plaid)
% plaid = BayesPlaid_sampleTau0(plaid)
%
% Sample Tau0, precision for the obervation function from
% fla Gamma prior assumption.
%
% Importance sampling with re-parameterization.
%
% input, output: plaid - MATLAB intance of simpler BayesPlaid model.
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last Update: 18/11/15 (dd/mm/yy)
EPS = 0.000001;
DEBUG = 0;

%% load variables

K = plaid.KK;
N1 = plaid.NN1;
N2 = plaid.NN2;

suf_r = plaid.suf_rr;
tau0 = plaid.ttau0;

test_matrix = plaid.test_matrix;

%% sampling

% new memory
new_tau0 = sampleTau(tau0, suf_r, test_matrix);
if( DEBUG )
    if(tau0 ~= new_tau0)
        display(['BayesPlaid_sampleTau0: tau0 changed ', num2str(tau0), ...
            ' -> ', num2str(new_tau0)]);
    end
    
end

%% save
plaid.ttau0 = new_tau0;

% end of sampleAB

%% internal functions

%% sample Tau
function new_tau = sampleTau(tau0, suf_r, test_matrix)

% pre-compute constant
sum_r = sum(sum( -0.5 * (1 - test_matrix) .* suf_r .* suf_r));
LL = sum(sum(test_matrix));
[N1 N2] = size(suf_r);

% Gamma hyper prior
gamma_a = 1.1;
gamma_lambda = 0.33; % larger expectation = larger variance
num_sample = 10;

% re-parameterize for uniform sampling
inv_values = 0.00001 + 0.99 * rand(1, num_sample);
gamma_values = [ (inv_values ./ (1 - inv_values)) tau0 ];

log_prob_gamma_values = zeros(size(gamma_values));
for ii=1:length(gamma_values)
    
    % prior
    log_prob_gamma_values(ii) = ...
        (gamma_a - 1) * log(gamma_values(ii)) ...
        - gamma_lambda * gamma_values(ii);
    
    % likelihood
    log_prob_gamma_values(ii) = log_prob_gamma_values(ii) ...
        + 0.5 * log(gamma_values(ii)) * (N1 * N2 - LL) ...
        + gamma_values(ii) * sum_r;
    
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
if new_tau > tau0 * 2
    new_tau = tau0 * 2;    
elseif new_tau < tau0 * 0.5
    new_tau = tau0 * 0.5;
end

