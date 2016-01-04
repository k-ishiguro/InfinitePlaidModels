function plaid = BayesPlaid_init(xx,test_index,KK,NN1,NN2,a_lambda0,b_lambda0,mu_theta0,tau_theta0,mu_phi0,tau_phi0,ttau00)
% plaid = BayesPlaid_init(xx,test_index,KK,NN1,NN2,..
%      a_lambda0,b_lambda0,mu_theta0,tau_theta0,mu_phi0,tau_phi0,tta00)
%
% initialize a simplified Bayes Plaid model for NEO bi-clustering. 
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last update 17/11/15 (ddmmyy)
DEBUG = 0;

%% constnats first
plaid.KK = KK;
plaid.NN1 = NN1;
plaid.NN2 = NN2;

%% hyperparameters
plaid.aa_lambda0 = a_lambda0;
plaid.bb_lambda0 = b_lambda0;
plaid.aa_lambda = ones(2,1) * a_lambda0;
plaid.bb_lambda = ones(2,1) * b_lambda0;

plaid.mmu_theta0 = mu_theta0;
plaid.ttau_theta0 = tau_theta0;
plaid.mmu_phi0 = mu_phi0;
plaid.ttau_phi0 = tau_phi0;

plaid.mmu_theta = mu_theta0;
plaid.ttau_theta = tau_theta0;
plaid.mmu_phi = mu_phi0;
plaid.ttau_phi = tau_phi0;

plaid.ttau00 = ttau00;
plaid.ttau0 = ttau00;

%% observation
plaid.xx = xx;

% masked elements indices (i,j)
plaid.test_index = test_index;

% quick reference for masks. if 1, then masked.
plaid.test_matrix = sparse(NN1, NN2);
for n=1:size(test_index, 1)
    ii = test_index(n,1);
    jj = test_index(n,2);
    plaid.test_matrix(ii, jj) = 1;
end

%% initialize hidden variables and parameters

% parameters
llambda1 = betarnd(ones(1, KK) * plaid.aa_lambda(1), ones(1, KK) * plaid.bb_lambda(1));
llambda2 = betarnd(ones(1, KK) * plaid.aa_lambda(2), ones(1, KK) * plaid.bb_lambda(2));
ttheta = plaid.mmu_theta + randn(KK, 1) / sqrt(plaid.ttau_theta);
pphi = plaid.mmu_phi + randn(1) / sqrt(plaid.ttau_phi);

% hidden variables. Random initialization with 1/0
zz1 = zeros(NN1,KK);
for i=1:NN1
    pp = llambda1 - rand(1, KK);
    idx = find(pp > 0);
    zz1(i, idx) = 1;
end

zz2 = zeros(NN2,KK);
for j=1:NN2
    pp = llambda2 - rand(1, KK);
    idx = find(pp > 0);
    zz2(j, idx) = 1;
end

for k=1:KK
zz1(k,k) = 1;
zz2(k,k) = 1;
end

plaid.llambda1 = llambda1;
plaid.llambda2 = llambda2;
plaid.ttheta = ttheta;
plaid.pphi = pphi;
plaid.zz1 = zz1;
plaid.zz2 = zz2;

%% count statistics: we always compute "full" counts. take care of test data when use these counts.
plaid.suf_mm = zeros(2, KK);
plaid.suf_mm(1, :) = sum(zz1, 1);
plaid.suf_mm(2, :) = sum(zz2, 1);

plaid.suf_MM = zeros(1, KK);
for k=1:KK
    zz_ij = zz1(:, k) * zz2(:, k)'; % is NN1 by NN2
    plaid.suf_MM(k) = sum(sum(zz_ij));
end

plaid.suf_rr = xx - pphi - zz1 * diag(ttheta) * zz2';

% Gibbs sample chace memories
plaid.pphi_gibbs = zeros(size(pphi));
plaid.ttheta_gibbs = zeros(size(ttheta));
plaid.llambda1_gibbs = zeros(size(llambda1));
plaid.llambda2_gibbs = zeros(size(llambda2));
plaid.zz1_gibbs = zeros(size(zz1));
plaid.zz2_gibbs = zeros(size(zz2));
