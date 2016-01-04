% Implemenation of a simplified Bayesian Plaid model for NEO bi-clsutering. .
%
% Reference: 
% Ishiguro, Sato, Nakano, Kimura, and Ueda, 
% ''Infinite Plaid Models for Infinite Bi-clustering", in AAAI, 2016. 
%
% This program pacakge is to demonstarte the concept of the paper, 
% and is NOT for full reproductions of expreimental results of the paper. 
%
% Be sure to read and agree the license file attached before using the program. 
% No comercial uses. 
% No distributions. 
% The author and the copyright holder have no responsible for any damages caused by the program, and have no responsible for future maintenance. 
%
% we solve by Partially Collapsed Gibbs. 
%
% Input data is assumed in the format of UCI-ML bag-of-words: 
% <top of the file>
% NN1 - number of domain 1 objects
% NN2 - number of domain 2 objects
% NNZ - number of nonzero lines = number of observations
% dom1-ID dom2-ID value 
% dom1-ID dom2-ID value  
% ...
% 
% BayesPlaid_run.m     - main test code. 
% 
% [Bayes Plaid model]
% pphi ~ Normal(mu_phi, tau_phi^{-1} )
% ttheta(k) ~ Normal(mu_theta, tau_theta^{-1})
% llambda1(k) ~ Beta(a_lambda(1), b_lambda(1))
% llambda2(k) ~ Beta(a_lambda(2), b_lambda(2))
% zz1(i,k) ~ Bernoulli(llambda1(k))
% zz2(j,k) ~ Bernoulli(llambda2(k))
% xx(i,j) ~ Normal( pphi + sum_{k} ttheta(k) * zz1(i,k) * zz2(i,k), ttau0^{-1} )
%
% [Observation]
% xx(i,j)           - NN1 by NN2 continuous valued matrix, possibly sparse. 
%                     Observations of relation 
%                     from domain 1 item i to domain 2 item j. 
% test_index(n, 2)  - n-th test (mask) data, row index(n,1) and column
%                     index(n,2)
% test_matrix(i,j)  - NN1 by NN2 sparse binary matrix. 
%                     if (i,j) = 1, then xx(i,j) is test data. 
%
% [current samples of variables]
% phi               - scaler, Normal mean parameter for non-interesting regions, or bias.  
% ttheta(k)         - KK array, Normal mean parameters for extracted submatrix k.
% lambda1(k)        - KK array, Beta distributed prob. of z1 = 1
% lambda2(k)        - KK array, Beta distributed prob. of z2 = 1
% zz1(i,k)          - NN1 by KK matrix, NN1 array of cluster
%                     assignments of domain 1, i-th item. 
% zz2(j,l)          - NN2 by KK matrix, NN2 array of cluster
%                     assignments of domein 2 j-th item. 
%
% [posteriors of variables]
% phi_gibbs         - scaler, Normal mean parameter for non-interesting regions, or bias.  
% ttheta_gibbs(k)   - KK array, Normal mean parameters for extracted submatrix k.
% lambda1_gibbs(k)  - KK array, Beta distributed prob. of z1 = 1
% lambda2_gibbs(k)  - KK array, Beta distributed prob. of z2 = 1
% zz1_gibbs(i,k)    - NN1 by KK matrix, NN1 array of cluster
%                     assignments of domain 1, i-th item. 
% zz2_gibbs(j,l)    - NN2 by KK matrix, NN2 array of cluster
%                     assignments of domein 2 j-th item. 
%
% [Constants]
% KK                - scaler, a number of cluster = submatrix. 
% NN1               - scaler, number of items in domain 1 (1 <= i <= NN1)
% NN2               - scaler, number of items in domain 2 (1 <= j <= NN2)
%
% [Count statistics]
% suf_mm(dom,k)     - 2 by KK array, counts of cluster assignments 
%                     mm(1,k) = \sum_{i} zz1(i,k), 
%                     mm(2,k) = \sum_{j} zz2(j,k)
% suf_MM(k)         - 1 by KK array, counts of pairwise cluster assignments 
%                     = \sum_{i,j} zz1(i,k) zz2(j,k)
% suf_rr(i,j)       - NN1 by NN2 array, residuals of factor models
%                     = xx(i,j) - phi 
%                       - \sum_{k=1}^{k} zz1(i,k) zz2(j,k) ttheta(k)
%
% [Hyper parameters]
% aa_lambda(dom)     - scalers, positive hyperparams for Beta shape parameters
% bb_lambda(dom)     - scalers, negative hyperparams for Beta shape parameters
% mmu_theta, ttau_theta - scalers, mean and precision of Normals for extracted submatrices.
% mmu_phi, ttau_phi   - scalers, mean and precision of Normal for not-extracted. 
% ttau0               - scaler, for precision of observation Gaussians
% aa_lambda0, bb_lambda0, mmu_theta0, ttau_theta0, mmu_phi0, ttau_phi0, ttau00 - initial values. 
%
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last update: 17/11/15 (dd/mm/yy)
