function InfinitePlaid_save(plaid,save_name_prefix,itr, ccc, SHOW_FIG, SAVE_FIG)
% InfinitePlaid_save(plaid,save_name_prefix,itr, ccc)
%
% Save the current data, figures and MAT of simpler Bayesian Plaid model.
%
% plaid             - MATLAB class instance, InfinitePlaid model.
% save_name_prefix  - string, prefix for files
% itr               - scaler, number of current interation.
% ccc               - 64 by 3 color map matrix.
% SHOW_FIG          - flag, if show figures, set 1.
% SAVE_FIG          - flag, if save figures, set 1.
%
% Written by Katsuhiko Ishiguro <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last update: 17/11/15 (dd/mm/yy)
figid = 10;

NN1 = plaid.NN1;
NN2 = plaid.NN2;

%% save the model in .mat
mat_name = [save_name_prefix, '_InfinitePlaid_itr', num2str(itr), '.mat'];
eval(['save ', mat_name, ' plaid save_name_prefix ccc itr -v7.3']);

if SHOW_FIG | SAVE_FIG
    
    %% submatrix means theta and background mean phi
    theta = plaid.ttheta;
    phi = plaid.pphi;
    figid = figid + 1;
    f = figure(figid);
    set(f, 'visible', 'off');
    bar([theta; phi])
    %colormap(jet)
    %colorbar
    ylabel('theta and phi');
    xlabel('submatrix index');
    
    if SHOW_FIG
        set(f, 'visible', 'on');
        drawnow;
    end
    
    if SAVE_FIG
        eval(['figname = ''', save_name_prefix, '_ThetaPhi'';']);
        eval(['figname_fig = ''', figname, '.fig'';']); saveas(gcf,figname_fig);
        eval(['figname_eps = ''', figname, '.eps'';']); saveas(gcf,figname_eps,'epsc');
        eval(['figname_png = ''', figname, '.png'';']); saveas(gcf,figname_png);
    end
    
    %% submatrix membership
    K = plaid.KK;
    N1 = plaid.NN1;
    N2 = plaid.NN2;
    z1 = plaid.zz1;
    z2 = plaid.zz2;
    
    % counts of all items
    mm_all = zeros(K+1, 1);
    for k=1:K
        mm_all(k) = sum(z1(:,k)) * sum(z2(:,k));
    end
    mm_all(K+1) = N1 * N2 - sum(mm_all(1:K));
    
    figid = figid + 1;
    f = figure(figid);
    set(f, 'visible', 'off');
    bar(mm_all);
    xlabel('submatrix index k');
    ylabel('membeship size');
    title(['Submatrix memberships at ', num2str(itr), '-th iteration']);
    
    if SHOW_FIG
        set(f, 'visible', 'on');
        drawnow;
    end
    
    if SAVE_FIG
        eval(['figname = ''', save_name_prefix, '_Memberships'';']);
        eval(['figname_fig = ''', figname, '.fig'';']); saveas(gcf,figname_fig);
        eval(['figname_eps = ''', figname, '.eps'';']); saveas(gcf,figname_eps,'epsc');
        eval(['figname_png = ''', figname, '.png'';']); saveas(gcf,figname_png);
    end
    
    %% all submatrices on the original permutation
    figid = figid + 1;
    f = figure(figid);
    set(f, 'visible', 'off');
    
    z_all = ones(N1, N2, 3);
    for k=1:K
        z_k = z1(:, k) * z2(:, k)';
        [i_idx, j_idx] = find(z_k > 0);
        for n=1:length(i_idx)
            ii = i_idx(n);
            jj = j_idx(n);
            z_all(ii,jj,:) = reshape(ccc(k,:),[1 1 3]);
            %z_all(ii, jj, :) = z_all(ii, jj, :) - z_k(ii,jj) * ( ones(1,1,3) - reshape(ccc(k,:), [1 1 3]) );
        end % end n-for
    end % end k-for
    image(z_all);
    %colormap( [ones(1, 3); ccc(k, :)] );
    xlabel('i')
    ylabel('j')
    title(['all submatrices'])
    
    if SHOW_FIG
        set(f, 'visible', 'on');
        drawnow;
    end
    
    if SAVE_FIG
        eval(['figname = ''', save_name_prefix, '_Z', ''';']);
        eval(['figname_fig = ''', figname, '.fig'';']); saveas(gcf,figname_fig);
        eval(['figname_eps = ''', figname, '.eps'';']); saveas(gcf,figname_eps,'epsc');
        eval(['figname_png = ''', figname, '.png'';']); saveas(gcf,figname_png);
    end
    
    %% sorted clustering on matrix
    
    for k=1:K
        
        % sort the item index
        z1k_idx = find( z1(:, k) == 1 )';
        z2k_idx = find( z2(:, k) == 1 )';
        z1k_diff_idx = setdiff(1:N1, z1k_idx);
        z2k_diff_idx = setdiff(1:N2, z2k_idx);
        
        sorted_item_idx1 = [z1k_idx z1k_diff_idx];
        sorted_item_idx2 = [z2k_idx z2k_diff_idx];
        
        % sort the observation
        xx_sorted = plaid.xx;
        xx_sorted = xx_sorted(sorted_item_idx1, :);
        xx_sorted = xx_sorted(:, sorted_item_idx2);
        
        figid = figid + 1;
        f = figure(figid);
        set(f, 'visible', 'off');
        imagesc(xx_sorted);
        colormap(1 - gray);
        
        xlabel('domain 1 i (sorted)');
        ylabel('domain 2 j (sorted)');
        
        % lines to separate the sub matrix
        
        col = ccc(k, :);
        hline1 = line( [0 length(z2k_idx)], [length(z1k_idx) length(z1k_idx)], 'LineWidth', 2, 'Color', col);
        hline2 = line( [length(z2k_idx) length(z2k_idx)], [0 length(z1k_idx)], 'LineWidth', 2, 'Color', col);
        
        if SHOW_FIG
            set(f, 'visible', 'on');
            drawnow;
        end
        
        
        if SAVE_FIG
            eval(['figname = ''', save_name_prefix, '_SortedEstimates', '_', num2str(k), ''';']);
            eval(['figname_fig = ''', figname, '.fig'';']); saveas(gcf,figname_fig);
            eval(['figname_eps = ''', figname, '.eps'';']); saveas(gcf,figname_eps,'epsc');
            eval(['figname_png = ''', figname, '.png'';']); saveas(gcf,figname_png);
        end
        
    end
    
end

