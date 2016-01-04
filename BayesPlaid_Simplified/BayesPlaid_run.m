% Test (demo) Code of a simplified Bayesian Plaid model for NEO bi-clutering 
% 
% Reference: 
% Ishiguro, Sato, Nakano, Kimura, and Ueda, 
% ''Infinite Plaid Models for Infinite Bi-clustering", in AAAI, 2016. 
%
% This program pacakge is to demonstarte the concept of the paper, 
% and is NOT for full reproductions of expreimental results of the paper. 
%
% Model: 
% x_{i,j} ~ Normal(phi + sum_{k} z_{1,i,k} z_{2,j,k} theta_{k}, tau0^{-1})
% z_{1,i,k} ~ Bernoulli(lambda_{1,k})
% lambda_{1,k} ~ Beta(a_{lambda}^{1}, b_{lambda}^{1})
% theta_{k} ~ Normal(mu_{theta}, tau_{theta}^{-1})
% phi ~ Norma(mu_{phi}, tau_{phi}^{-1})
%
% We employ a Partially Collapsed Gibbs.
% Marginalize lambda out, keep thtea and phi for easy implementation.
%
% Input data: assumed in the format of UCI-ML bag-of-words:
% <top of the file>
% NN1 - number of domain 1 objects
% NN2 - number of domain 2 objects
% NNZ - number of nonzero lines = number of observations
% dom1ID dom2ID value
% dom1ID dom2ID value
% ...
%
% Hyper parameters you need to specify first:
% KK                      - scaler, number of submatrix
% a_lambda0, b_lambda0    - scalars for Beta of lambda for 2 domains
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
SHOW_FIG = 1; % 1 if showing figures
SAVE_FIG = 1; % 1 if saveing figures

HYPRM_UPDATE = 1% 1 if update hyperparameters

% number of sub matrix
KK = 5 % 3

% hyper parameters
a_lambda0 = [1.0, 1.0];
b_lambda0 = [9.0, 9.0];
mu_theta0 = 4.0;
tau_theta0 = 10.0;
mu_phi0 = 0.05
tau_phi0 = 10.0;
tau0 = 1.0

% output prefix
save_name_prefix = '../testBayesPlaid';

BurnInItr = 10
MaxItr = 30
MCMCItrGap = 5

% observation
%file_name = '../data/synth2_X.dat';
file_name = '../data/synth2_X.mat';

% answer
Z_file_name = '../data/synth2_Z.mat';

save_name_prefix = [save_name_prefix, ...
    '_K', num2str(KK), ...
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
plaid = BayesPlaid_init(xx,test_index,KK,NN1,NN2,a_lambda0,b_lambda0,mu_theta0,tau_theta0,mu_phi0,tau_phi0,tau0);

% recorder
recorder = BayesPlaid_recorder_init(ccc, MaxItr);
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
    former_z1 = plaid.zz1;
    former_z2 = plaid.zz2;
    
    start_time = tic;
    
    plaid = BayesPlaid_sampleZ(plaid);
    
    % parameters that are NOT marginalized out
    plaid = BayesPlaid_sampleTheta(plaid);
    plaid = BayesPlaid_samplePhi(plaid);
    
    % these parameters are marginalized out during the inference
    plaid = BayesPlaid_sampleLambda(plaid); % compute MAP lambda
    
    if( HYPRM_UPDATE )
        plaid = BayesPlaid_sampleABLambda(plaid);
        plaid = BayesPlaid_sampleMuTauTheta(plaid);
        plaid = BayesPlaid_sampleMuTauPhi(plaid);
        plaid = BayesPlaid_sampleTau0(plaid);
     end
   
    % MCMC posterior cache
    if(itr > BurnInItr)
        if( mod( itr - BurnInItr, MCMCItrGap ) == 0)
            mcmc_itr = (itr - BurnInItr)/ MCMCItrGap;
            
            plaid = BayesPlaid_cacheMCMC(plaid, mcmc_itr);
        end
    end
    
    % record the current update evaluations
    recorder = BayesPlaid_record(plaid, recorder, itr, Z_file_name,SHOW_FIG);
    
    % print iteration information
    dur_time = toc(start_time);
    BayesPlaid_printCurrentStatus(plaid,recorder,itr,dur_time,ccc,SHOW_FIG);
    
end % end itr-for

disp('Gibbs finished\n');
%% output

% save the model, displays
BayesPlaid_save(plaid, save_name_prefix, itr, ccc, SHOW_FIG, SAVE_FIG);

% save the recorder
BayesPlaid_saveRecord(plaid, recorder, save_name_prefix, itr, Z_file_name, SHOW_FIG, SAVE_FIG);
