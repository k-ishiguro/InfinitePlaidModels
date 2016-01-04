function plaid = BayesPlaid_sampleABLambda(plaid)
% plaid = BayesPlaid_sampleABLambda(plaid)
%
% Sample A and B, for sampling lambda of cluster assignment parameters from
% flat Gamma prior assumption.
%
% Importance sampling with re-parameterization.
%
% input, output: plaid - MATLAB intance of simpler BayesPlaid model.
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last Update: 17/11/15 (dd/mm/yy)
EPS = 0.000001;
DEBUG = 0;

%% load variables

K = plaid.KK;
N1 = plaid.NN1;
N2 = plaid.NN2;

a_lambda = plaid.aa_lambda;
b_lambda = plaid.bb_lambda;

suf_m = plaid.suf_mm;

%% sampling

% new memory
new_a_lambda = ones(size(a_lambda));
new_b_lambda = ones(size(b_lambda));

% for the first domain
new_a_lambda(1) = sampleA(a_lambda(1), b_lambda(1), K, N1, suf_m(1, :));
new_b_lambda(1) = sampleB(a_lambda(1), b_lambda(1), K, N1, suf_m(1, :));
if( DEBUG )
    if(a_lambda(1) ~= new_a_lambda(1))
        display(['BayesPlaid_sampleABLambda: domain 1 a_lambda changed ', num2str(a_lambda(1)), ...
            ' -> ', num2str(new_a_lambda(1))]);
    end
    
    if(b_lambda(1) ~= new_b_lambda(1))
        display(['BayesPlaid_sampleABLambda: domain 1 b_lambda changed ', num2str(b_lambda(1)), ...
            ' -> ', num2str(new_b_lambda(1))]);
    end
end

new_a_lambda(2) = sampleA(a_lambda(2), b_lambda(2), K, N2, suf_m(2, :));
new_b_lambda(2) = sampleB(a_lambda(2), b_lambda(2), K, N2, suf_m(2, :));
if( DEBUG )
    if(a_lambda(2) ~= new_a_lambda(2))
        display(['BayesPlaid_sampleABLambda: domain 2 a_lambda changed ', num2str(a_lambda(2)), ...
            ' -> ', num2str(new_a_lambda(2))]);
    end
    
    if(b_lambda(2) ~= new_b_lambda(2))
        display(['BayesPlaid_sampleABLambda: domain 2 b_lambda changed ', num2str(b_lambda(2)), ...
            ' -> ', num2str(new_b_lambda(2))]);
    end
end

%% save
plaid.aa_lambda = new_a_lambda;
plaid.bb_lambda = new_b_lambda;

% end of sampleAB

%% internal functions
function new_a = sampleA(a, b, K, N, suf_m)

% Gamma hyper prior
gamma_a = 1.1;
gamma_lambda = 0.33; % larger expectation = larger variance
num_sample = 10;

% re-parameterize for uniform sampling
inv_values = 0.00001 + 0.99 * rand(1, num_sample);
gamma_values = [ 1.0 + (inv_values ./ (1 - inv_values)) a ];

log_prob_gamma_values = zeros(size(gamma_values));

for ii=1:length(gamma_values)
    
    % prior
    log_prob_gamma_values(ii) = ...
        (gamma_a - 1) * log(gamma_values(ii)) ...
        - gamma_lambda * gamma_values(ii);
    
    % likelihood
    log_prob_gamma_values(ii) = log_prob_gamma_values(ii) ...
        + K * gammaln(gamma_values(ii) + b) ...
        - K * gammaln(gamma_values(ii));
    
    for k=1:K
        log_prob_gamma_values(ii) = log_prob_gamma_values(ii) ...
            + gammaln(gamma_values(ii) + suf_m(k)) ...
            - gammaln(gamma_values(ii) + b + N);
    end % k-for
    
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

new_a = gamma_values(ss);
% do not allow large moves
if new_a > a * 2
    new_a = a * 2;    
elseif new_a < a * 0.5
    new_a = a * 0.5;
end

%% sample b
function new_b = sampleB(a, b, K, N, suf_m)

% Gamma hyper prior
gamma_a = 1.1;
gamma_lambda = 0.33; % larger expectation = larger variance
num_sample = 10;

% re-parameterize for uniform sampling
inv_values = 0.00001 + 0.99 * rand(1, num_sample);
gamma_values = [ 1.0 + (inv_values ./ (1 - inv_values)) b ];


log_prob_gamma_values = zeros(size(gamma_values));
for ii=1:length(gamma_values)
    
    % prior
    log_prob_gamma_values(ii) = ...
        (gamma_a - 1) * log(gamma_values(ii)) ...
        - gamma_lambda * gamma_values(ii);
    
    % likelihood
    log_prob_gamma_values(ii) = log_prob_gamma_values(ii) ...
        + K * gammaln(a + gamma_values(ii)) ...
        - K * gammaln(gamma_values(ii));
    
    for k=1:K
        log_prob_gamma_values(ii) = log_prob_gamma_values(ii) ...
            + gammaln(gamma_values(ii) + N - suf_m(k)) ...
            - gammaln(a + gamma_values(ii) + N);
    end % k-for
    
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

new_b = gamma_values(ss);
% do not allow large moves
if new_b > b * 2
    new_b = b * 2;    
elseif new_b < b * 0.5
    new_b = b * 0.5;
end
