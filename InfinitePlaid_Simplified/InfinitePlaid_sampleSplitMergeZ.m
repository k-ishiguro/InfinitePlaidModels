function plaid = InfinitePlaid_sampleSplitMergeZ(plaid)
% plaid = InfinitePlaid_sampleSplitMerge(plaid)
%
% Perform Split-merge Metropolis hasting for Infinite Plaid model.
%
% This is crucial to compute i and j, mixed and randomly.
%
% input:
% plaid       - MATLAB intance of InfinitePlaid model.
%
% output:
% plaid       - MATLAB intance of InfinitePlaid model, updated
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last Updated: 17/11/15 (dd/mm/yy)
DEBUG = 0; % set 2 if deep debug
EPS = 0.0000001;

times = zeros(1,8);

%% load variables

K = plaid.KK;
N1 = plaid.NN1;
N2 = plaid.NN2;

% if no cluster, skip
if K == 0
    return
end

% for observation paraeter, no marginalization.
phi = plaid.pphi;
theta = plaid.ttheta;
tau0 = plaid.ttau0;

% for z assignments, marginalize lambda.
ggamma = plaid.ggamma;

% to draw new dishes
mu_theta = plaid.mmu_theta;
tau_theta = plaid.ttau_theta;

x = plaid.xx;
test_matrix = plaid.test_matrix;
%test_index = plaid.test_index;

z1 = plaid.zz1;
z2 = plaid.zz2;

% counts
suf_m = plaid.suf_mm;
suf_M = plaid.suf_MM;
suf_r = plaid.suf_rr;
suf_r = (1.0 - test_matrix) .* suf_r;

% we also need Gibbs cache memories
K_gibbs = plaid.KK_gibbs;
theta_gibbs = plaid.ttheta_gibbs;
z1_gibbs = plaid.zz1_gibbs;
z2_gibbs = plaid.zz2_gibbs;

%% proposal anchors

% pick two anchors based on areas

area_k = zeros(1,K);
for k=1:K
    area_k(k) = sum(z1(:,k)) * sum(z2(:,k));
end
area_k = area_k + EPS;
pp = area_k / sum(area_k); % sum to one

uu = rand;
c_1 = 1+sum(uu>cumsum(pp));
assert(c_1 < K+2);
assert(c_1 > 0);

uu = rand;
c_2 = 1+sum(uu>cumsum(pp));
assert(c_2 < K+2);
assert(c_2 > 0);

%% MERGE move
if c_1 ~= c_2
    
    z1_12 = z1(:, c_1) + z1(:, c_2);
    z1_merge = zeros(N1, 1);
    z1_merge(z1_12 > 0) = 1;
    
    z2_12 = z2(:, c_1) + z2(:, c_2);
    z2_merge = zeros(N2, 1);
    z2_merge(z2_12 > 0) = 1;
    
    % prior log ratio
    prior_log_ratio = -1.0 * sum(log(ggamma));
    prior_log_ratio = prior_log_ratio ...
        - gammaln(N1 - sum(z1(:, c_1)) + 1) ...
        - gammaln(sum(z1(:, c_1))) ...
        - gammaln(N1 - sum(z1(:, c_2)) + 1) ...
        - gammaln(sum(z1(:, c_2))) ...
        + gammaln(N1+1);
    prior_log_ratio = prior_log_ratio ...
        - gammaln(N2 - sum(z2(:, c_1)) + 1) ...
        - gammaln(sum(z2(:, c_1))) ...
        - gammaln(N2 - sum(z2(:, c_2)) + 1) ...
        - gammaln(sum(z2(:, c_2))) ...
        + gammaln(N2+1);
    prior_log_ratio = prior_log_ratio ...
        + gammaln(N1 - sum(z1_merge) + 1) ...
        + gammaln(sum(z1_merge)) ...
        + gammaln(N2 - sum(z2_merge) + 1) ...
        + gammaln(sum(z2_merge));
    
    % transition log ratio
    trans_log_ratio = -1 * (sum(z1_merge) + sum(z2_merge) - 2) * log(2);
    
    % likelihood
    i_merge_idx = find(z1_merge > 0);
    j_merge_idx = find(z2_merge > 0);
    
    % for split (original)
    r_ij = suf_r(i_merge_idx, :);
    r_ij = r_ij(:, j_merge_idx);
    errs = -0.5 * tau0 * r_ij .* r_ij;
    lk_log_ratio = -1 * sum(sum(errs));
    
    % for merge (new)
    suf_r_merge = suf_r;
    suf_r_merge = suf_r_merge + theta(c_1) * z1(:, c_1) * z2(:, c_1)';
    suf_r_merge = suf_r_merge + theta(c_2) * z1(:, c_2) * z2(:, c_2)';
    r_ij = suf_r_merge(i_merge_idx, :);
    r_ij = r_ij(:, j_merge_idx);
    
    new_theta = (theta(c_1) + theta(c_2)) / 2; % you may sample from the posterior
    r_ij = r_ij - new_theta;
    errs = -0.5 * tau0 * r_ij .* r_ij;
    lk_log_ratio = lk_log_ratio + sum(sum(errs));
    
    % test
    accept_rate = min(1, exp(trans_log_ratio + prior_log_ratio + lk_log_ratio));
    if DEBUG==2
        prior_log_ratio
        trans_log_ratio
        lk_log_ratio
        accept_rate
    end
    if rand(1) < accept_rate
        
        % merge move
        new_c = min(c_1, c_2);
        out_c = max(c_1, c_2);
        z1(:, new_c) = z1_merge;
        z2(:, new_c) = z2_merge;
        
        theta(new_c) = new_theta;
        
        % deletion
        theta(out_c) = [];
        z1(:, out_c) = [];
        z2(:, out_c) = [];
        
        %theta_gibbs(out_c) = [];
        %z1_gibbs(:, out_c) = [];
        %z2_gibbs(:, out_c) = [];
        
        suf_m(:, out_c) = [];
        suf_M(out_c) = [];
        % suf_r: effect is computed already
        
        K = K - 1;
        %K_gibbs = K_gibbs - 1;
        
        % re-compute all sufficient statistics
        suf_m(1,:) = max(sum(z1, 1), EPS);
        suf_m(2,:) = max(sum(z2, 1), EPS);
        for k=1:K
            suf_M(k) = max(sum(z1(:,k)) * sum(z2(:,k)), EPS);
        end % end k-for
        suf_r = x - phi - z1 * diag(theta) * z2';
        
        if DEBUG
            fprintf(1, ['InfinitePlaid_sampleSplitMergeZ: merge move accepted. sub-matrices deleted, KK=', num2str(K), '\n']);        end
        
    end % end accept-if0
    
    %% split move
elseif c_1 == c_2
    
    % first generate a new factor
    z1_split1 = zeros(N1, 1);
    z1_split2 = zeros(N1, 1);
    z2_split1 = zeros(N2, 1);
    z2_split2 = zeros(N2, 1);
    
    c1_idx = z1(:, c_1) > 0;
    for ii=1:length(c1_idx)        
        i_now = c1_idx(ii);        
        if ii==1
            z1_split1(i_now) = 1;
        elseif ii==2
            z1_split2(i_now) = 1;
        else
            if rand(1) > 0.5
                z1_split1(i_now) = 1;
            else
                z1_split2(i_now) = 1;
            end
        end
    end
    
    c2_idx = z2(:, c_1) > 0;
    for ii=1:length(c2_idx)       
        i_now = c2_idx(ii);        
        if ii==1
            z2_split1(i_now) = 1;
        elseif ii==2
            z2_split2(i_now) = 1;
        else
            if rand(1) > 0.5
                z2_split1(i_now) = 1;
            else
                z2_split2(i_now) = 1;
            end
        end
    end
    
    % prior log ratio
    prior_log_ratio = 1.0 * log(ggamma);
    prior_log_ratio = prior_log_ratio ...
        + gammaln(N1 - sum(z1_split1) + 1) ...
        + gammaln(sum(z1_split1)) ...
        + gammaln(N1 - sum(z1_split2) + 1) ...
        + gammaln(sum(z1_split2)) ...
        - gammaln(N1+1);
    prior_log_ratio = prior_log_ratio ...
        + gammaln(N2 - sum(z2_split1) + 1) ...
        + gammaln(sum(z2_split1)) ...
        + gammaln(N2 - sum(z2_split2) + 1) ...
        + gammaln(sum(z2_split2)) ...
        - gammaln(N2+1);
    prior_log_ratio = prior_log_ratio ...
        - gammaln(N1 - sum(z1(:, c_1)) + 1) ...
        - gammaln(sum(z1(:, c_1))) ...
        - gammaln(N2 - sum(z2(:, c_2)) + 1) ...
        - gammaln(sum(z2(:, c_2)));
    
    % transition log ratio
    trans_log_ratio = (sum(z1(:, c_1)) + sum(z2(:, c_1)) - 2) * log(2);
    
    % likelihood
    i_merge_idx = find(z1(:, c_1) > 0);
    j_merge_idx = find(z2(:, c_1) > 0);
    
    % for merge (original)
    r_ij = suf_r(i_merge_idx, :);
    r_ij = r_ij(:, j_merge_idx);
    errs = -0.5 * tau0 * r_ij .* r_ij;
    lk_log_ratio = -1 * sum(sum(errs));
    
    % for split (new)
    suf_r_split = suf_r;
    suf_r_split = suf_r_split + theta(c_1) * z1(:, c_1) * z2(:, c_1)';
    
    new_theta1 = theta(c_1); % you may sample from the posterior
    new_theta2 = theta(c_1);
    suf_r_split = suf_r_split - new_theta1 * z1_split1 * z2_split1';
    suf_r_split = suf_r_split - new_theta2 * z1_split2 * z2_split2';
    
    r_ij = suf_r_split(i_merge_idx, :);
    r_ij = r_ij(:, j_merge_idx);
    errs = -0.5 * tau0 * r_ij .* r_ij;
    lk_log_ratio = lk_log_ratio + sum(sum(errs));
    
    % test
    accept_rate = min(1, exp(trans_log_ratio + prior_log_ratio + lk_log_ratio));
        if DEBUG==2
        prior_log_ratio
        trans_log_ratio
        lk_log_ratio
        accept_rate
    end
    if rand(1) < accept_rate
        
        % split move
        z1(:, c_1) = z1_split1;
        z2(:, c_1) = z2_split1;
        
        theta(c_1) = new_theta1;
        
        % addition
        theta = [theta; new_theta2];
        z1 = [z1 z1_split2];
        z2 = [z2 z2_split2];
        
        %theta_gibbs = [theta_gibbs; zeros(1)];
        %z1_gibbs = [z1_gibbs zeros(N1, 1)];
        %z2_gibbs = [z2_gibbs zeros(N2, 1)];
        
        K = K + 1;
        %K_gibbs = K_gibbs + 1;
        
        % re-compute all sufficient statistics
        suf_m = zeros(2, K);
        suf_m(1,:) = max(sum(z1, 1), EPS);
        suf_m(2,:) = max(sum(z2, 1), EPS);
        for k=1:K
            suf_M(k) = max(sum(z1(:,k)) * sum(z2(:,k)), EPS);
        end % end k-for
        suf_r = x - phi - z1 * diag(theta) * z2';
        
        if DEBUG
            fprintf(1, ['InfinitePlaid_sampleSplitMergeZ: split move accepted. sub-matrices added, KK=', num2str(K), '\n']);
        end
        
    end % end accept-if
    
end

%% save
plaid.KK = K;
plaid.zz1 = z1;
plaid.zz2 = z2;
plaid.ttheta = theta;

plaid.KK_gibbs = K_gibbs;
plaid.ttheta_gibbs = theta_gibbs;
plaid.zz1_gibbs = z1_gibbs;
plaid.zz2_gibbs = z2_gibbs;

plaid.suf_mm = suf_m;
plaid.suf_MM = suf_M;
plaid.suf_rr = suf_r;