function val = computeMyNMIOverlap(Z, trueZ)
% val = computeMyNMIOverlap(z, treuZ)
%
% Compute the Normalized Mutual information extended for overlapping clustering (Lancichinetti 2009).  
% This does not reduce to the "Original" NMI if no overlap. 
%
% input:
% Z          - K cell array, clustered object indices. 
% trueZ      - trueK cell array, the ground truth clustering indices. 
%
% Written by Katsuhiko Ishiguro <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last update: 08/01/15 (dd/mm/yy)
EPS = 0.00000001;

K = length(Z);
trueK = length(trueZ);

mem1 = Z{1};
for k=2:K
    mem1 = union(mem1, Z{k});
end
mem2 = trueZ{1};
for j=2:trueK
    mem2 = union(mem2, trueZ{j});
end
assert( length(mem1) == length(mem2) );
N = length(mem1);

% compute empirical membership prob. and entropy
p_Zk = zeros(K, 2);
for k=1:K
    p_Zk(k, 1) = length(Z{k}) / N;
    p_Zk(k, 2) = 1 - length(Z{k}) / N;
end

p_tZj = zeros(trueK, 2);
for j=1:trueK
    p_tZj(j, 1) = length(trueZ{j}) / N;
    p_tZj(j, 2) = 1 - length(trueZ{j}) / N;
end

H_Zk = zeros(K,1);
EPS_idx = find(p_Zk < EPS);
large_idx = find(p_Zk > 1.0 );
log_p = log(p_Zk);
log_p(EPS_idx) = 0;
log_p(large_idx) = 0;
for k=1:K
   H_Zk(k) = -1.0 * sum(p_Zk(k,:) .* log_p(k,:)); 
end
EPS_idx = find(H_Zk < EPS);
H_Zk(EPS_idx) = EPS;

H_tZj = zeros(trueK,1);
EPS_idx = find(p_tZj < EPS);
large_idx = find(p_tZj > 1.0 );
log_p = log(p_tZj);
log_p(EPS_idx) = 0;
log_p(large_idx) = 0;
for j=1:trueK
    H_tZj(j) = -1.0 * sum(p_tZj(j,:) .* log_p(j,:));
end
EPS_idx = find(H_tZj < EPS);
H_tZj(EPS_idx) = EPS;

% compute empirical joint membership prob. and entropy
p_kj = zeros(4, K, trueK);
for k=1:K
    for j=1:trueK
        joints = length(intersect(Z{k}, trueZ{j}));
        p_kj(1, k,j) = joints / N;
        
        p_kj(2,k,j) = (length(Z{k}) - joints) / N;
        p_kj(3,k,j) = (length(trueZ{j}) - joints) / N;
        
        p_kj(4,k,j) = (N - length(Z{k}) - length(trueZ{j}) + joints) / N;
        
    end
end

H_kj = zeros(K, trueK);
EPS_idx = find(p_kj < EPS);
large_idx = find(p_kj > 1.0 );
p_kj(large_idx) = 1.0;
log_p = log(p_kj);
log_p(EPS_idx) = 0;
log_p(large_idx) = 0;
for k=1:K
    for j=1:trueK
        H_kj(k,j) = -1.0 * sum(p_kj(:, k,j) .* log_p(:,k,j));       
    end
end

% compute NMI
H_kj1 = H_kj - repmat(H_tZj, [1 K])';
H_k = min(H_kj1, [], 2);

H_kj2 = H_kj - repmat(H_Zk, [1 trueK]);
H_j = min(H_kj2, [], 1);

H1 = sum(H_k ./ H_Zk) / K;
H2 = sum(H_j' ./ H_tZj) / trueK;

val = 1.0 - ((H1 + H2) / 2.0);

%if val < 0.0
%    val = 0.0;
%elseif val > 1.0
%    val = 1.0;
%end



