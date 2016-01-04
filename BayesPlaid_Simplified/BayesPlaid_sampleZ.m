function plaid = BayesPlaid_sampleZ(plaid)
% plaid = BayesPlaid_sampleZ(plaid)
%
% Sample Z1 and Z2, submatrix assignment variables,
% for simpler Bayesian Plaid model.
% This is crucial to compute i and j, mixed and randomly.
%
% input:
% plaid       - MATLAB intance of BayesPlaid model.
%
% output:
% plaid       - MATLAB intance of BayesPlaid model, updated
%
% Written by Katsuhiko ISHIGURO <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last Updated: 17/11/15 (dd/mm/yy)
DEBUG = 0;
EPS = 0.0000001;

times = zeros(1,5);

%% load variables

K = plaid.KK;
N1 = plaid.NN1;
N2 = plaid.NN2;

% for observation paraeter, no marginalization.
phi = plaid.pphi;
theta = plaid.ttheta;
tau0 = plaid.ttau0;

% for z assignments, marginalize lambda.
a_lambda = plaid.aa_lambda;
b_lambda = plaid.bb_lambda;

x = plaid.xx;
test_matrix = plaid.test_matrix;
%test_index = plaid.test_index;

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
        %% domain 1
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
            
            % log ratio
            pp = 0;
            % prior
            Gibbs_start = tic;
            
            pp = pp + log(a_lambda(1) + suf_m(1,k));
            pp = pp - log(b_lambda(1) + N1 - suf_m(1,k));
            
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
            
            pp = exp(pp) / (exp(pp) + 1); % sum to one
            uu = rand;
            if(uu < pp)
                z_val = 1;
            else
                z_val = 0;
            end % end if-else
            
            if(DEBUG)
                fprintf(1, ['BayesPlaid_sampleZ:domain 1, i=', num2str(i), ', k=', num2str(k), ', z(i,k)=', num2str(z_val), ' has sampled \n']);
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
        
    else
        %% domain 2
        
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
            pp = 0;
            % prior
            Gibbs_start = tic;
            
            pp = pp + log(a_lambda(2) + suf_m(2,k));
            pp = pp - log(b_lambda(2) + N2 - suf_m(2,k));
            
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
            
            pp = exp(pp) / (exp(pp) + 1); % sum to one
            uu = rand;
            if(uu < pp)
                z_val = 1;
            else
                z_val = 0;
            end % end if-else
            
            if(DEBUG)
                fprintf(1, ['BayesPlaid_sampleZ:domain 2, j=', num2str(j), ', k=', num2str(k), ', z(j,k)=', num2str(z_val), ' has sampled \n']);
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
plaid.zz1 = z1;
plaid.zz2 = z2;
plaid.suf_mm = suf_m;
plaid.suf_MM = suf_M;
plaid.suf_rr = suf_r;

if DEBUG
    times
end
