function plaid = InfinitePlaid_init(xx,test_index,initKK,NN1,NN2,ggamma10, ggamma20,mu_theta0,tau_theta0,mu_phi0,tau_phi0,ttau00)
% plaid = InfinitePlaid_init(xx,test_index,initKK,NN1,NN2,..
%      ggamma10,ggamma20,mu_theta0,tau_theta0,mu_phi0,tau_phi0,tta00)
%
% initialize a Simplified Infinite Plaid model for Infinite bi-clustering
% collapsed Gibbs solution.
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last update 17/11/15 (ddmmyy)
DEBUG = 0;

%% constnats first
plaid.initKK = initKK;
plaid.KK = initKK;
plaid.NN1 = NN1;
plaid.NN2 = NN2;

% harmonic is in symbolic math. we define below
plaid.HarmonicNum = [my_harmonic(NN1) my_harmonic(NN2)];

%% hyperparameters
plaid.ggamma10 = ggamma10;
plaid.ggamma20 = ggamma20;
plaid.ggamma = [ggamma10 ggamma20];

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
v1 = betarnd(ones(1,initKK) * plaid.ggamma(1), ones(1,initKK));
v2 = betarnd(ones(1,initKK) * plaid.ggamma(2), ones(1,initKK));
llambda1 = ones(1,initKK);
llambda2 = ones(1,initKK);
for k=1:initKK
   llambda1(k:end) = llambda1(k:end) * v1(k); 
   llambda2(k:end) = llambda1(k:end) * v2(k);
end

ttheta = plaid.mmu_theta + randn(initKK, 1) / sqrt(plaid.ttau_theta);
pphi = plaid.mmu_phi + randn(1) / sqrt(plaid.ttau_phi);

% hidden variables. Random initialization with 1/0
zz1 = zeros(NN1,initKK);
for i=1:NN1
    pp = llambda1 - rand(1, initKK);
    idx = find(pp > 0);
    zz1(i, idx) = 1;
end

zz2 = zeros(NN2,initKK);
for j=1:NN2
    pp = llambda2 - rand(1, initKK);
    idx = find(pp > 0);
    zz2(j, idx) = 1;
end

% force assginemnts at least one entries per sub-matrix
for k=1:initKK
    zz1(k,k) = 1;
    zz2(k,k) = 1;
end

plaid.ttheta = ttheta;
plaid.pphi = pphi;
plaid.zz1 = zz1;
plaid.zz2 = zz2;

%% count statistics: we always compute "full" counts. take care of test data when use these counts.
plaid.suf_mm = zeros(2, initKK);
plaid.suf_mm(1, :) = sum(zz1, 1);
plaid.suf_mm(2, :) = sum(zz2, 1);

plaid.suf_MM = zeros(1, initKK);
for k=1:initKK
    zz_ij = zz1(:, k) * zz2(:, k)'; % is NN1 by NN2
    plaid.suf_MM(k) = sum(sum(zz_ij));
end

plaid.suf_rr = xx - pphi - zz1 * diag(ttheta) * zz2';

% Gibbs sample chace memories
plaid.KK_gibbs = initKK;
plaid.pphi_gibbs = zeros(size(pphi));
plaid.ttheta_gibbs = zeros(size(ttheta));
plaid.zz1_gibbs = zeros(size(zz1));
plaid.zz2_gibbs = zeros(size(zz2));

function Hn = my_harmonic(n)
% compute the harmonic function of n, 
% \sum_{i=1}^{n} 1/i
Hn = 0;
for i=1:n
    Hn = Hn + 1.0 / i;
end


