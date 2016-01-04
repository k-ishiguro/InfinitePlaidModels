% Test (demo) Code of a Simplified Infinite Plaid model for Infinite Bi-clustering
%
% Reference: 
% Ishiguro, Sato, Nakano, Kimura, and Ueda, 
% ''Infinite Plaid Models for Infinite Bi-clustering", in AAAI, 2016. 
%
% This program pacakge is to demonstarte the concept of the paper, 
% and is NOT for full reproductions of expreimental results of the paper. 
%
% Model: 
% pphi ~ Normal(mu_phi, tau_phi^{-1} )
% ttheta(k) ~ Normal(mu_theta, tau_theta^{-1})
% v1_{l} ~ Beta(ggamma(1), 1)
% v2_{l} ~ Beta(ggamma(2), 1)
% llambda1(k) = \prod_{l=1}^{k} v_{1,l}
% llambda2(k) = \prod_{l=1}^{k} v_{2,l}
% zz1(i,k) ~ Bernoulli(llambda1(k))
% zz2(j,k) ~ Bernoulli(llambda2(k))
% xx(i,j) ~ Normal( pphi + sum_{k} ttheta(k) * zz1(i,k) * zz2(i,k), ttau0^{-1} )
%
%
% We employ a Partially Collapsed Gibbs.
% Marginalize v (lambda) out, keep thtea and phi for easy implementation.
%
% Input data is assumed in the format of UCI-ML bag-of-words:
% <top of the file>
% NN1 - number of domain 1 objects
% NN2 - number of domain 2 objects
% NNZ - number of nonzero lines = number of observations
% dom1ID dom2ID value
% dom1ID dom2ID value
% ...
%
% Hyper parameters you need to specify first:
% init                    - scaler, number of submatrix to be initialized
% ggamma1, ggamma2      - scalars for IBP conecntration parameters
% mu_theta0, tau_theta0   - scalars for Normal of thetas
% mu_phi0, tau_phi0       - scalars for Normal of phi
% tau0                    - scaler, precision for observation variances.
%
% BurnInItr      - maximum iterations for burn-in processes
% MaxItr         - maximum iterations for sampling processes
% MCMCItrGap        - sampling gap for Gibbs.
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last Update: 18/11/15 (dd/mm/yy)
clear all; close all; c = clock; rand(floor(sum(c(1:6))));

figid = 0;
DEBUG = 0;
SHOW_FIG = 1; % if 1 showing figures
SAVE_FIG = 0; % if 1 saving figures

HYPRM_UPDATE = 1 % if 1 updates

% number of sub matrix
initKK = 10 % 3

% hyper parameters
ggamma10 = 0.1;
ggamma20 = 0.1;
mu_theta0 = 4.0;
tau_theta0 = 10.0;
mu_phi0 = 0.05;
tau_phi0 = 10.0;
tau0 = 1.0;

% output prefix
save_name_prefix = '../testInfinitePlaid';

BurnInItr = 10
MaxItr = 30
MCMCItrGap = 5

file_name = '../data/synth2_X.mat';
%file_name = '../data/synth2_X.dat';
Z_file_name = '../data/synth2_Z.mat';

save_name_prefix = [save_name_prefix, ...
    '_K', num2str(initKK), ...
    '_itr', num2str(MaxItr), ...
    '_mutheta', num2str(mu_theta0), ...
    '_muphi', num2str(mu_phi0), ...
    '_tau0', num2str(tau0), ...
    ];
if(HYPRM_UPDATE)
    save_name_prefix = [save_name_prefix, ...
        '_hyprm', ...,
        ];
end

save_name_prefix = [save_name_prefix, '_XXX']

%% input the data

% observation: UCI-ML BoW format only
%[xx NN1 NN2] = readUCIBoW_sparse(file_name);
load(file_name);
NN1 = N1;
NN2 = N2;

% random masks for test-data evaluations.
ratio_flips = 0.01;
test_i_index = [];
test_j_index = [];
for i=1:NN1
    rand_vec = rand(NN2, 1);
    test_j_index_i = find(rand_vec < ratio_flips);
    test_i_index_i = i * ones( size(test_j_index_i) );
    
    test_i_index = [test_i_index; test_i_index_i];
    test_j_index = [test_j_index; test_j_index_i];
end
test_index = [test_i_index test_j_index];

fprintf('loading data done\n');
%% set up the model

% color array
% initialize colors to be used for plotting
ccc = colormap;
ccc = ccc(randperm(size(ccc,1)),:);
close all;

% initialize the model
plaid = InfinitePlaid_init(xx,test_index,initKK,NN1,NN2,ggamma10,ggamma20,mu_theta0,tau_theta0,mu_phi0,tau_phi0,tau0);

% recorder
recorder = InfinitePlaid_recorder_init(ccc, MaxItr);
record_name = [save_name_prefix, '_record'];

% show observation
if SHOW_FIG
    figid = figid + 1;
    figure(figid);
    imagesc(xx);
    colormap(1-gray);
    colorbar;
    xlabel('1st domain i');
    ylabel('2nd domain j');
    title('observation');
end

fprintf('initialize done\n');

%% Inference

% for stopping
for itr=1:MaxItr

    % main target: hidden variables for submatrix assignments
    %former_z1 = plaid.zz1;
    %former_z2 = plaid.zz2;
    
    start_time = tic;
   
    plaid = InfinitePlaid_sampleZ(plaid);
    plaid = InfinitePlaid_sampleSplitMergeZ(plaid);
   
    % parameters that are NOT marginalized out
        plaid = InfinitePlaid_sampleTheta(plaid);
    plaid = InfinitePlaid_samplePhi(plaid);
       
    if( HYPRM_UPDATE )
        
        plaid = InfinitePlaid_sampleGamma(plaid);
        plaid = InfinitePlaid_sampleMuTauTheta(plaid);
        plaid = InfinitePlaid_sampleMuTauPhi(plaid);
        plaid = InfinitePlaid_sampleTau0(plaid);
            end
     
    % MCMC posterior cache
    if(itr > BurnInItr)
        if( mod( itr - BurnInItr, MCMCItrGap ) == 0)
            mcmc_itr = (itr - BurnInItr)/ MCMCItrGap;
            
            plaid = InfinitePlaid_cacheMCMC(plaid, mcmc_itr);
        end
    end
    
    % record the current update evaluations
    recorder = InfinitePlaid_record(plaid, recorder, itr, Z_file_name,SHOW_FIG);
    
    % print iteration information
    dur_time = toc(start_time);
    InfinitePlaid_printCurrentStatus(plaid,recorder,itr,dur_time,ccc,SHOW_FIG);
        
end % end itr-for

disp('Gibbs finished: final sweep');

% final sweep to clean small guys
if(1)
    plaid = InfinitePlaid_sweepZ(plaid);
    plaid = InfinitePlaid_sampleTheta(plaid);
    plaid = InfinitePlaid_samplePhi(plaid);
    
    if( HYPRM_UPDATE )
        plaid = InfinitePlaid_sampleGamma(plaid);
        plaid = InfinitePlaid_sampleMuTauTheta(plaid);
        plaid = InfinitePlaid_sampleMuTauPhi(plaid);
        plaid = InfinitePlaid_sampleTau0(plaid);
    end

    if(itr > BurnInItr)
        if( mod( itr - BurnInItr, MCMCItrGap ) == 0)
            mcmc_itr = (itr - BurnInItr)/ MCMCItrGap;
            
            plaid = InfinitePlaid_cacheMCMC(plaid, mcmc_itr);
        end
    end
    
    % record the current update evaluations
    recorder = InfinitePlaid_record(plaid, recorder, itr, Z_file_name,SHOW_FIG);
    
    InfinitePlaid_printCurrentStatus(plaid,recorder,itr,dur_time,ccc,SHOW_FIG);
    
end % end if

%% output

% save the model, displays
InfinitePlaid_save(plaid, save_name_prefix, itr, ccc, SHOW_FIG, SAVE_FIG);

% save the recorder
InfinitePlaid_saveRecord(plaid, recorder, save_name_prefix, itr, Z_file_name, SHOW_FIG, SAVE_FIG);
