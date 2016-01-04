function plaid = InfinitePlaid_sampleTheta(plaid)
% plaid = InfinitePlaid_sampleTheta(plaid)
%
% Sample Theta, cluster observation parameters,
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

times = zeros(1,5);

%% load variables

K = plaid.KK;

% for observation paraeter, no marginalization.
theta = plaid.ttheta;
tau0 = plaid.ttau0;

mu_theta = plaid.mmu_theta;
tau_theta = plaid.ttau_theta;

%x = plaid.xx;
test_matrix = plaid.test_matrix;

z1 = plaid.zz1;
z2 = plaid.zz2;

% counts
suf_r = plaid.suf_rr;

%% Gibbs sampling, cluster-wise

permed = randperm(K);
%permed = 1:NN;

for kk=1:K
    Gibbs_start = tic;
    
    k = permed(kk);
    z_responsible = z1(:,k) * z2(:,k)'; % N1 by N2 binary matrix
    
    if(DEBUG)
        display(['InfinitePlaid_sampleTheta @ k=', num2str(k)])
        display(['number of active entries=' num2str(sum(sum(z_responsible)))])
    end
    
    % if the cluster is empty, update from the prior
    if( sum(sum(z_responsible)) < EPS )
        if(DEBUG)
            display(['(almost) no entries; pass'])
            display(['\n'])
        end
        
        k_val = mu_theta + randn(1) / sqrt(tau_theta);
        theta(k) = k_val;
        
        continue
    end
    
    %% remove the parameter
    k_val = theta(k);
    
    % decrease counts
    suf_r = suf_r + z_responsible * k_val;
    
    if(DEBUG)
        display(['sum of absolute residuals=', num2str(sum(sum(abs(suf_r))))])
        display(['average of absolute residuals=', num2str(sum(sum(abs(suf_r))) / sum(sum(z_responsible))) ])
    end
    
    times(1) = times(1) + toc(Gibbs_start);
    
    %% compute the required quantities for posterior distribution
    Gibbs_start = tic;
    
    z_responsible_train = z_responsible - test_matrix;
    z_responsible_train_idx = find(z_responsible_train > 0); % positive assignments && not test entries
    
    Nk = length(z_responsible_train_idx);
    y_k = suf_r(z_responsible_train_idx);
    y_bark = sum(y_k) / max(Nk, EPS);
    
    if(DEBUG)
        display(['number of active training entries=', num2str(Nk)])
        display(['sum of absolute residuals in training=', num2str(sum(sum(abs(y_k))))])
        display(['average of absolute residuals=', num2str(sum(sum(abs(y_k))) / max(Nk, EPS) ) ])
        display(['\n'])
    end
    
    times(2) = times(2) + toc(Gibbs_start);
    
    %% Sampling from the posterior
    Gibbs_start = tic;
    
    new_tau = max( EPS, tau_theta + Nk * tau0 ); % for numerical stability
    new_mean = (tau_theta * mu_theta + Nk * tau0 * y_bark) / new_tau;
    new_var = 1.0 / new_tau;
    
    k_val = new_mean + randn(1) * sqrt(new_var);
    %k_val = new_mean; % MAP or ML value
        
    %% store the new parameter
    theta(k) = k_val;
    
    % increase counts
    suf_r = suf_r - z_responsible * k_val;
    times(3) = times(3) + toc(Gibbs_start);
    
end % end index-permuated-for

%% save
plaid.ttheta = theta;
plaid.suf_rr = suf_r;

if DEBUG
    display(['Computational Time for SampleTheta:'])
    display(times)
end
