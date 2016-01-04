function plaid = InfinitePlaid_sampleGamma(plaid)
% plaid = InfinitePlaid_sampleGamma(plaid)
%
% Sample Gamma, the IBP concentration parameter for cluster assignments Z. 
% Hyper prior is assumed to be flat Gamma. 
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
N1 = plaid.NN1;
N2 = plaid.NN2;

ggamma = plaid.ggamma;
harm_num = plaid.HarmonicNum;

%% sampling

% new memory
new_ggamma = ones(size(ggamma));

new_ggamma(1) = sampleAlpha(ggamma(1), K, harm_num(1));
if( DEBUG )
    if(ggamma(1) ~= new_ggamma(1))
        display(['InfinitePlaid_sampleGamma: domain 1 ggamma changed ', num2str(ggamma(1)), ...
            ' -> ', num2str(new_ggamma(1))]);
    end
end

new_ggamma(2) = sampleAlpha(ggamma(2), K, harm_num(2));
if( DEBUG )
    if(ggamma(2) ~= new_ggamma(2))
        display(['InfinitePlaid_sampleGamma: domain 2 ggamma changed ', num2str(ggamma(2)), ...
            ' -> ', num2str(new_ggamma(2))]);
    end
end

%% save
plaid.ggamma = new_ggamma;

% end of sampleGamma

%% internal functions
function new_alpha = sampleAlpha(alpha, K, Hn)

% Gamma hyper prior
gamma_a = 1.1;
gamma_lambda = 0.3; % for alpha, we don't want large values
num_sample = 10;

% re-parameterize for uniform sampling
inv_values = 0.00001 + 0.99 * rand(1, num_sample);
gamma_values = [ 1.0 + (inv_values ./ (1 - inv_values)) alpha ];

log_prob_gamma_values = zeros(size(gamma_values));

for ii=1:length(gamma_values)
    
    % prior
    log_prob_gamma_values(ii) = ...
        (gamma_a - 1) * log(gamma_values(ii)) ...
        - gamma_lambda * gamma_values(ii);
    
    % likelihood
    log_prob_gamma_values(ii) = log_prob_gamma_values(ii) ...
        + K * log(gamma_values(ii)) ...
        - Hn * gamma_values(ii);
    
end % end ii-for
log_prob_gamma_values = log_prob_gamma_values - max(log_prob_gamma_values);

% sample
pp = exp(log_prob_gamma_values);
pp = pp / sum(pp);

rr = rand;
ss = 1 + sum(rr > cumsum(pp));
if(ss > num_sample + 1)
    ss = num_sample + 1;
end

new_alpha = gamma_values(ss);
% do not allow large moves
if new_alpha > alpha * 2
    new_alpha = alpha * 2;    
elseif new_alpha < alpha * 0.5
    new_alpha = alpha * 0.5;
end


