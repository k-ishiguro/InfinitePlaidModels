function plaid = InfinitePlaid_samplePhi(plaid)
% plaid = InfinitePlaid_samplePhi(plaid)
%
% Sample Phi, background observation parameters,
% for Infinite Plaid model.
%
% input:
% plaid       - MATLAB intance of InfinitePlaid model.
%
% output:
% plaid       - MATLAB intance of InfinitePlaid model, updated
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last Updated: 17/11/15 (dd/mm/yy)
DEBUG = 0;
EPS = 0.000001;

%% load variables

% for observation paraeter, no marginalization.
phi = plaid.pphi;
tau0 = plaid.ttau0;

mu_phi = plaid.mmu_phi;
tau_phi = plaid.ttau_phi;

test_matrix = plaid.test_matrix;

% counts
suf_r = plaid.suf_rr;

%% Posterior sampling

% decrease counts
suf_r = suf_r + phi;

% compute the required quantities for posterior distribution
y = suf_r; % including test entries
responsible_train_idx = find(test_matrix < 1); % not test entries

N = length(responsible_train_idx);
%N = N1 * N2 - length(responsible_train_idx);
y = y(responsible_train_idx);
y_bar = sum(y) / N;

% Sampling from the posterior
new_tau = max( EPS, tau_phi + N * tau0 ); % for numerical stability
new_mean = (tau_phi * mu_phi + N * tau0 * y_bar) / new_tau;
new_var = 1.0 / new_tau;

phi = new_mean + randn(1) * sqrt(new_var);

% increase counts
suf_r = suf_r - phi;

%% save
plaid.pphi = phi;
plaid.suf_rr = suf_r;

if DEBUG
    times
end
