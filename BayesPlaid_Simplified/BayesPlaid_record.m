function recorder = BayesPlaid_record(plaid, recorder, recitr, Z_file_name,SHOW_FIG)
% recorder = BayesPlaid_record(plaid, recorder, recitr)
%
% record the current estimate results and some other staitstics.
%
% input:
% plaid        - MATLAB class instance, BayesPlaid model.
% recorder     - MATALB class instacne, recording component.
% recitr       - scaler, number of current recording iterations (not the
%                true iteration)
% Z_file_name - String, the path to 'true' submatrix assignments
% SHOWFIG       - 1/0, flag for figure display.
%
% output:
% recorder  - overwritten recording component
%
% Written by Katsuhiko Ishiguro <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last update: 18/11/15 (dd/mm/yy)
DEBUG = 0;

%% get current evaluations' histories
loglk = recorder.loglk;
loglk_marginal = recorder.loglk_marginal;

test_loglk = recorder.test_loglk;
test_loglk_marginal = recorder.test_loglk_marginal;

if ~strcmp(Z_file_name, 'null')
    nmi_history = recorder.nmi_history;
end

%% data to load

test_matrix = plaid.test_matrix;
test_index = plaid.test_index;
LL = size(test_index, 1);

K = plaid.KK;
N1 = plaid.NN1;
N2 = plaid.NN2;

x = plaid.xx;

phi = plaid.pphi;
theta = plaid.ttheta;
tau0 = plaid.ttau0;

z1 = plaid.zz1;
z2 = plaid.zz2;

suf_r = plaid.suf_rr;

phi_gibbs = plaid.pphi_gibbs;
theta_gibbs = plaid.ttheta_gibbs;
z1_gibbs = plaid.zz1_gibbs;
z2_gibbs = plaid.zz2_gibbs;

% precompute constant
log_const = -0.5 * log(2.0 * pi) + 0.5 * log(tau0);

%% averaged log likelihood

% log likelihood under current samples
loglk(recitr) = 0;
test_loglk(recitr) = 0;

% full error terms (inside exp)
errs = -0.5 * tau0 * suf_r .* suf_r;

train_errs = (1.0 - test_matrix) .* errs;
test_errs = test_matrix .* errs;

loglk(recitr) = log_const + sum(sum(train_errs)) / (N1 * N2 - LL);
test_loglk(recitr) = log_const + sum(sum(test_errs)) / LL;

%% averaged Gibbs-marginalized log likelihood

% Compute the observation parameters
% using expectation of Z and expectation of theta, phi
suf_r_gibbs = x - phi_gibbs - z1_gibbs * diag(theta_gibbs) * z2_gibbs';

% log likelihood under Gibbs-marginalized expectations of variables
loglk_marginal(recitr) = 0;
test_loglk_marginal(recitr) = 0;

% full error terms (inside exp)
errs = -0.5 * tau0 * suf_r_gibbs .* suf_r_gibbs;

train_errs = (1.0 - test_matrix) .* errs;
test_errs = test_matrix .* errs;

loglk_marginal(recitr) = log_const + sum(sum(train_errs)) / (N1 * N2 - LL);
test_loglk_marginal(recitr) = log_const + sum(sum(test_errs)) / LL;

%% Compute averages of NMI if the answer is provided
if ~strcmp(Z_file_name, 'null')
    load(Z_file_name);
    trueK = K;
    K = plaid.KK; % reload
    true_z1 = zz1;
    true_z2 = zz2;
    
    % we need to count the number of "entries"
    true_assigns = cell(1, trueK+1);
    assigns = cell(1, K+1);
    negatives = ones(N1, N2);
    for k=1:trueK
        temp = true_z1(:, k) * true_z2(:, k)';
        true_assigns{k} = find(temp > 0);
        negatives = negatives - temp;
    end
    true_assigns{trueK+1} = find(negatives > 0);
    
    negatives = ones(N1, N2);
    for k=1:K
        temp = z1(:, k) * z2(:, k)';
        assigns{k} = find(temp > 0);
        negatives = negatives - temp;
    end
    assigns{K+1} = find(negatives > 0);
    
    % NMI
    nmi_history(recitr) = computeMyNMIOverlap(assigns, true_assigns);
    
end

%% show evolutions
if SHOW_FIG
    fig = figure(100);
    subplot(4,1,1)
    plot(1:recitr, loglk(1:recitr));
    title('Ave. Training Log Lk')
    
    subplot(4,1,2)
    plot(1:recitr, test_loglk(1:recitr));
    title('Ave Test Log Lk');
    
    subplot(4,1,3)
    plot(1:recitr, loglk_marginal(1:recitr));
    title('Ave. Marginalized Training Log Lk')
    
    subplot(4,1,4)
    plot(1:recitr, test_loglk_marginal(1:recitr));
    title('Ave Marginalized Test Log Lk');
    
    if ~strcmp(Z_file_name, 'null')
        fig = figure(101);
        plot(1:recitr, nmi_history(1:recitr));
        title('NMI');
    end
    drawnow;
end
%% save
recorder.loglk = loglk;
recorder.loglk_marginal = loglk_marginal;

recorder.test_loglk = test_loglk;
recorder.test_loglk_marginal = test_loglk_marginal;

if ~strcmp(Z_file_name, 'null')
       recorder.nmi_history = nmi_history;
end
