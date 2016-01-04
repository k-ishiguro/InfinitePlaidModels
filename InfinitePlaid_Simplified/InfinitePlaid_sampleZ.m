function plaid = InfinitePlaid_sampleZ(plaid)
% plaid = InfinitePlaid_sampleZ(plaid)
%
% Sample Z1 and Z2, submatrix assignment variables,
% for Simplified Infinite Plaid model.
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

z1 = plaid.zz1;
z2 = plaid.zz2;

% counts
suf_m = plaid.suf_mm;
suf_M = plaid.suf_MM;
suf_r = plaid.suf_rr;

%% Gibbs sampling, object-wise

permed = randperm(N1 + N2);
%permed = 1:NN;

for nn=1:(N1+N2)
    
    i = permed(nn);
    flag = 1;
    
    if(i > N1)
        j = i - N1;
        flag = 0;
    end
    
    if(flag)
        %
        %% domain 1. start from existing sub-matrices, then sample new dishes
        %
        
        %
        % Existing sub-matrices
        %
        
        % choose k to remove
        perm_k = randperm(K);
        for kk=1:K
            k = perm_k(kk);
            
            %% remove the assignments
            z_val = z1(i,k);
            z1(i,k) = -1;
            Gibbs_start = tic;
            % decrease counts
            suf_m(1,k) = suf_m(1,k) - z_val;
            if z_val==1
                suf_M(k) = suf_M(k) - sum(z2(:,k));
                for jj=1:N2
                    suf_r(i,jj) = suf_r(i,jj) + z2(jj,k) * theta(k);
                end % end jj-for
            end % z_val-if
            
            times(1) = times(1) + toc(Gibbs_start);
            
            %% compute the log ratio of assignment probability of being "1"
            
            Gibbs_start = tic;
            % log ratio
            pp = computeLogPrior(suf_m(1,k), N1, EPS);
            
            times(2) = times(2) + toc(Gibbs_start);
            
            % likelihoods
            Gibbs_start = tic;
            for jj=1:N2
                if test_matrix(i,jj)==0
                    jj_term = z2(jj,k) * theta(k);
                    jj_term = jj_term * (theta(k) - 2.0 * suf_r(i,jj));
                    pp = pp - 0.5 * tau0 * jj_term;
                end % end test_matrix-if
            end % end jj-for
            times(3) = times(3) + toc(Gibbs_start);
            
            %% sample the assignment by posterior
            Gibbs_start = tic;
            z_val = sampleBernoulli( pp );
            if(DEBUG == 2)
                fprintf(1, ['InfinitePlaid_sampleZ:domain 1, KK=', num2str(K), ', i=', num2str(i), ', k=', num2str(k), ', z(i,k)=', num2str(z_val), ' has sampled. \n']);
            end
            
            %% add the new assignment and counts
            z1(i,k) = z_val;
            
            % increment counts
            suf_m(1,k) = suf_m(1,k) + z_val;
            if z_val==1
                suf_M(k) = suf_M(k) + sum(z2(:,k));
                for jj=1:N2
                    suf_r(i,jj) = suf_r(i,jj) - z2(jj,k) * theta(k);
                end % end jj-for
            end % z_val-if
            
            times(4) = times(4) + toc(Gibbs_start);
        end % end kk-for
        
        %
        % Remove unrepresented sub-matrices (dishes) here.
        %
        Gibbs_start = tic;
        for k=K:-1:1
            if (suf_m(1,k) < 1)
                
                if K > 1
                    theta(k) = [];
                    z1(:, k) = [];
                    z2(:, k) = [];
                    
                    suf_m(:, k) = [];
                    suf_M(k) = [];
                    % suf_r: effect is computed already
                    
                elseif K==1
                    theta(1) = 0;
                    z1(:, 1) = 0;
                    z2(:, 1) = 0;
                    
                    suf_m(:, 1) = 0;
                    suf_M(1) = 0;
                else
                    % cannot happen
                    assert(1==0);
                end
                
                K = K - 1;
                
                if DEBUG
                    fprintf(1, ['InfinitePlaid_sampleZ: domain 1 sub-matrix', num2str(k), ' has deleted, KK=', num2str(K), '\n']);
                end
            end
        end % end k-for
        
        %
        % Metropolis-Hasting move for new dish draws
        %
        k_new = poissrnd(ggamma(1) / N1);
        if k_new > 0 % propose new dishes
            
            % assignments
            z1_l = zeros(N1, k_new);
            z1_l(i, :) = 1;
            
            z2_l = zeros(N2, k_new);
            rand_z2l = rand(N2, k_new);
            z2_l( rand_z2l < ggamma(2) / (ggamma(2) + N2) ) = 1;
            % to assure assignment
            if ( sum(z2_l) == 0 )
                z2_l(1) = 1;
            end
            
            % parameters
            theta_l = mu_theta + randn(k_new, 1) / sqrt(tau_theta);
            
            % residuals
            yj = z2_l * theta_l;
            
            % acceptance test
            acceptance_ratio = exp( max( 0, -0.5 * tau0 * yj .* (yj - 2 * suf_r(i,:)') ) );
            if rand(1) < acceptance_ratio
                %
                % Add new sub-marices (dishes) here
                %
                
                if K > 0
                    z1 = [z1 z1_l];
                    z2 = [z2 z2_l];
                    theta = [theta; theta_l];
                    
                elseif K == 0
                    z1 = z1_l;
                    z2 = z2_l;
                    theta = theta_l;
                else
                    % won't happen
                    assert(1==0);
                end
                
                
                K = K + k_new;
                
                suf_m = zeros(2, K);
                suf_m(1,:) = max(sum(z1, 1), EPS);
                suf_m(2,:) = max(sum(z2, 1), EPS);
                for k=1:K
                    suf_M(k) = max(sum(z1(:,k)) * sum(z2(:,k)), EPS);
                end % end k-for
                suf_r = x - phi - z1 * diag(theta) * z2';
                
                if DEBUG
                    fprintf(1, ['InfinitePlaid_sampleZ: domain 1 MH step accepted. sub-matrix', num2str(k_new), ' sub-matrices added, KK=', num2str(K), '\n']);
                end
            else
                if DEBUG==2
                    fprintf(1, ['InfinitePlaid_sampleZ: domain 1 MH step rejected. \n']);
                end
                
            end % end acceptance_ratio-ifelse
            
        else
            if DEBUG==2
                fprintf(1, ['InfinitePlaid_sampleZ: domain 1 MH step rejected. \n']);
            end
        end % end k_new-ifelse
        times(5) = times(5) + toc(Gibbs_start);
        
    else
        %
        %% domain 2. start from existing sub-matrices, then sample new dishes
        %
        
        %
        % Existing sub-matrices
        %
        
        % choose k to remove
        perm_k = randperm(K);
        for kk=1:K
            k = perm_k(kk);
            
            %% remove the assignments
            z_val = z2(j,k);
            z2(j,k) = -1;
            Gibbs_start = tic;
            % decrease counts
            suf_m(2,k) = suf_m(2,k) - z_val;
            if z_val==1
                suf_M(k) = suf_M(k) - sum(z1(:,k));
                for ii=1:N1
                    suf_r(ii,j) = suf_r(ii,j) + z1(ii,k) * theta(k);
                end % end jj-for
            end % z_val-if
            times(1) = times(1) + toc(Gibbs_start);
            
            %% compute the log ratio of assignment probability of being "1"
            
            % log ratio
            Gibbs_start = tic;
            pp = computeLogPrior(suf_m(2,k), N2, EPS);
            
            times(2) = times(2) + toc(Gibbs_start);
            
            % likelihoods
            Gibbs_start = tic;
            for ii=1:N1
                if test_matrix(ii,j)==0
                    ii_term = z1(ii,k) * theta(k);
                    ii_term = ii_term * (theta(k) - 2.0 * suf_r(ii,j));
                    pp = pp - 0.5 * tau0 * ii_term;
                end % end test_matrix-if
                
            end % end ii-for
            times(3) = times(3) + toc(Gibbs_start);
            
            %% sample the assignment by posterior
            Gibbs_start = tic;
            z_val = sampleBernoulli( pp );
            
            if(DEBUG == 2)
                fprintf(1, ['InfinitePlaid_sampleZ:domain 2, KK=', num2str(K), ', j=', num2str(j), ', k=', num2str(k), ', z(j,k)=', num2str(z_val), ' has sampled \n']);
            end
            
            %% add the new assignment and counts
            z2(j,k) = z_val;
            
            % increment counts
            suf_m(2,k) = suf_m(2,k) + z2(j,k);
            if z_val==1
                suf_M(k) = suf_M(k) + sum(z1(:,k));
                for ii=1:N1
                    suf_r(ii,j) = suf_r(ii,j) - z1(ii,k) * theta(k);
                end % end ii-for
            end % z_val-if
            
            times(4) = times(4) + toc(Gibbs_start);
            
        end % end kk-for
        
        %
        % Remove unrepresented sub-matrices (dishes) here.
        %
        Gibbs_start = tic;
        for k=K:-1:1
            if (suf_m(2,k) < 1)
                
                if K > 1
                    theta(k) = [];
                    z1(:, k) = [];
                    z2(:, k) = [];
                    
                    suf_m(:, k) = [];
                    suf_M(k) = [];
                    % suf_r: effect is computed already
                elseif K==1
                    theta(1) = 0;
                    z1(:, 1) = 0;
                    z2(:, 1) = 0;
                    
                    suf_m(:, 1) = 0;
                    suf_M(1) = 0;
                    % suf_r: effect is computed already
                else
                    % won't happen
                    assert(1==0);
                end
                
                K = K - 1;
                %K_gibbs = K_gibbs - 1;
                
                if DEBUG
                    fprintf(1, ['InfinitePlaid_sampleZ: domain 2 sub-matrix', num2str(k), ' has deleted, KK=', num2str(K), '\n']);
                end
            end
        end % end k-for
        
        %
        % Metropolis-Hasting move for new dish draws
        %
        k_new = poissrnd(ggamma(2) / N2);
        if k_new > 0 % propose new dishes
            
            % assignments
            z2_l = zeros(N2, k_new);
            z2_l(j, :) = 1;
            
            z1_l = zeros(N1, k_new);
            rand_z1l = rand(N1, k_new);
            z1_l( rand_z1l < (ggamma(1) / (ggamma(1) + N1)) ) = 1;
            % to assure assignment
            if ( sum(z1_l) == 0 )
                z1_l(1) = 1;
            end
            
            % parameters
            theta_l = mu_theta + randn(k_new, 1) / sqrt(tau_theta);
            
            % residuals
            yi = z1_l * theta_l;
            
            % acceptance test
            acceptance_ratio = exp( max( 0, -0.5 * tau0 * yi .* (yi - 2 * suf_r(:, j)) ) );
            if rand(1) < acceptance_ratio
                %
                % Add new sub-marices (dishes) here
                %
                
                if K > 0
                    z1 = [z1 z1_l];
                    z2 = [z2 z2_l];
                    theta = [theta; theta_l];
                elseif K == 0
                    z1 = z1_l;
                    z2 = z2_l;
                    theta = theta_l;
                else
                    % won't happen
                    assert(1==0);
                end
                
                K = K + k_new;
                %K_gibbs = K_gibbs + k_new;
                
                %suf_m_temp = suf_m;
                suf_m = zeros(2, K);
                suf_m(1,:) = max(sum(z1, 1), EPS);
                suf_m(2,:) = max(sum(z2, 1), EPS);
                for k=1:K
                    suf_M(k) = max(sum(z1(:,k)) * sum(z2(:,k)), EPS);
                end % end k-for
                suf_r = x - phi - z1 * diag(theta) * z2';
                
                if DEBUG
                    fprintf(1, ['InfinitePlaid_sampleZ: domain 2 MH step accepted. sub-matrix', num2str(k_new), ' sub-matrices added, KK=', num2str(K), '\n']);
                end
                
            else
                
                if DEBUG==2
                    fprintf(1, ['InfinitePlaid_sampleZ: domain 2 MH step rejected. \n']);
                end
                
            end % end acceptance_ratio-ifelse
        else
            if DEBUG==2
                fprintf(1, ['InfinitePlaid_sampleZ: domain 2 MH step rejected. \n']);
            end
            
        end % end k_new-ifelse
        times(5) = times(5) + toc(Gibbs_start);
        
    end % end domain-flag-ifelse
    
end % end index-permuated-for


%% recompute counts for safety
suf_m(1,:) = max(sum(z1, 1), EPS);
suf_m(2,:) = max(sum(z2, 1), EPS);
for k=1:K
    suf_M(k) = max(sum(z1(:,k)) * sum(z2(:,k)), EPS);
end % end k-for
suf_r = x - phi - z1 * diag(theta) * z2';

%% save
plaid.KK = K;
plaid.zz1 = z1;
plaid.zz2 = z2;
plaid.ttheta = theta;

%plaid.KK_gibbs = K_gibbs;
%plaid.ttheta_gibbs = theta_gibbs;
%plaid.zz1_gibbs = z1_gibbs;
%plaid.zz2_gibbs = z2_gibbs;

plaid.suf_mm = suf_m;
plaid.suf_MM = suf_M;
plaid.suf_rr = suf_r;

if DEBUG==2
    times
end

%% inernal functions

function pp = computeLogPrior(suf_m, N, EPS)
% compute the log ratio of prior of existing sub-matrix

numer = max(suf_m, 0);
denom = max(N - suf_m, EPS);
pp = log( numer / denom );
% end of computeLogPrior

function z = sampleBernoulli( log_pp )
% sample from Bernoulli when log ratio of being 1 is log_pp

pp = exp(log_pp) / (exp(log_pp) + 1); % sum to one
uu = rand;
if(uu < pp)
    z = 1;
else
    z = 0;
end % end if-else
% end of sampleBernoulli
