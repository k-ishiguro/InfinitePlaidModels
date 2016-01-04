function recorder = BayesPlaid_recorder_init(cc,TotalSteps)
% recorder = BayesPlaid_recorder_init(cc,TotalSteps,Z_file_name)
%
% set up the recorder components for simpler Bayesian Plaid model. 
%
% input: 
% cc          - matrix, color array
% TotalSteps  - Number of recording times if we loop unil reaces last. 
%
% output: 
% recorder    - recording component. 
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last update: 17/11/15 (dd/mm/yy)


% history of averaged training data log likelihood on current variable sample
loglk = zeros(1, TotalSteps);

% history of averaged training data Gibbs-marginalized log likelihood
loglk_marginal = zeros(1, TotalSteps);

% history of averaged test data log likelihood on current variable sample
test_loglk = zeros(1, TotalSteps);

% history of averaged test data Gibbs-marginalized log likeihood
test_loglk_marginal = zeros(1, TotalSteps);

% history of normalized mutual information for overlaps
nmi_history = zeros(1, TotalSteps);
    
%% save all
recorder.cc = cc;
recorder.TotalSteps = TotalSteps;

recorder.loglk = loglk;
recorder.loglk_marginal = loglk_marginal;

recorder.test_loglk = test_loglk;
recorder.test_loglk_marginal = test_loglk_marginal;

recorder.nmi_history = nmi_history;
