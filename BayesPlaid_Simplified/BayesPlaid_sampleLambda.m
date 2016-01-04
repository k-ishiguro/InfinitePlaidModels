function plaid = BayesPlaid_sampleLambda(plaid)
% plaid = BayesPlaid_sampleLambda(plaid)
%
% Sample Lambda, Beta parmaeter for cluster assignments, 
% for simpler Bayesian Plaid model.
%
% input:
% plaid       - MATLAB intance of BayesPlaid model.
%
% output:
% plaid       - MATLAB intance of BayesPlaid model, updated
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last Updated: 17/11/15 (dd/mm/yy)
DEBUG = 0;
EPS = 0.000001;

%% load variables

N1 = plaid.NN1;
N2 = plaid.NN2;

a_lambda = plaid.aa_lambda + EPS;
b_lambda = plaid.bb_lambda + EPS;

% counts
suf_m = plaid.suf_mm;

%% Easy posterior sampling and save

plaid.llambda1 = betarnd(a_lambda(1) + suf_m(1,:), b_lambda(1) + N1 - suf_m(1,:));
plaid.llambda2 = betarnd(a_lambda(2) + suf_m(2,:), b_lambda(2) + N2 - suf_m(2,:));

if DEBUG
    times
end
