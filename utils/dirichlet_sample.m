function r = dirichlet_sample(a,n)
% r = dirichlet_samle(a,n)
% 
% draw n samples of dirichlet distributions, which have a parameter vector
% of a. 
% 
% a    - 1 by d vector of dirichlet parameter
% n    - number of samples
% r    - n by d matrix, n samples of d-dimensional row vector, each sampled
% from dirichlet(a). 
% 
% Written by Katsuhiko ISHIGURO <ishiguro@cslab.kecl.ntt.co.jp>
% May 12, 2009. 
%
[k, d] = size(a);
assert(k == 1);

r = zeros(n, d);
for i=1:n
    rn = gamrnd(a,ones(1,d));
    sum_rn = sum(rn);
    r(i, :) = rn / sum_rn;
    
%     % force sum to 1 (for computational precision matter)
%     sum_rn = sum( r(i, :) );
%     gap = sum_rn - 1;
%     if(gap < 0)
%         r(n, 1) = r(n, 1) - gap;
%     elseif(gap > 0)
%         [max_r max_i] = max(r(i, :));
%         r(i, max_i) = r(i, max_i) - gap;
%     end
    
end
