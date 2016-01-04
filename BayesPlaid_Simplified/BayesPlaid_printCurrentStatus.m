function BayesPlaid_printCurrentStatus(plaid, recorder, itr, time_elapsed, ccc, SHOW_FIG)
% BayesPlaid_printCurrentStatus(plaid, recorder, itr, time_elapsed,cc,SHOWFIG)
%
% print the current monitor status.
%
% plaid         - MATLAB class intance, BayesPlaid model.
% recorder      - recorder instances, stores history of evaluations.
% itr           - scaler, number of current iteration.
% time_elapsed  - scaler, computational time for this iteration.
% ccc           - 64 by 3 color map matrix.
% SHOW_FIG       - 1/0, flag for figure display.
%
% Written by Katushiko Ishgiuro <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last update: 05/01/15
figid = 10;
DEBUG = 0;
SHOW_CLUSTER_FIGS = 1;

%% print the summry on the console

fprintf(1, ['iter=', num2str(itr), ...
    ' time=', num2str(time_elapsed), ...
    ' train loglk=', num2str(recorder.loglk(itr)), ...
    ' test loglk=', num2str(recorder.test_loglk(itr)), ...
    '\n']);

%% figures
if(SHOW_FIG)
    
    K = plaid.KK;
    N1 = plaid.NN1;
    N2 = plaid.NN2;
    
    z1 = plaid.zz1;
    z2 = plaid.zz2;
    %x = plaid.xx;
    
    %% submatrix membership
    
    % counts of all items
    mm_all = zeros(K+1, 1);
    for k=1:K
        mm_all(k) = sum(z1(:,k)) * sum(z2(:,k));
    end
    mm_all(K+1) = N1 * N2 - sum(mm_all(1:K));
    
    figid = figid + 1;
    figure(figid);
    bar(mm_all);
    xlabel('submatrix index k');
    ylabel('membeship size');
    title(['Submatrix memberships at ', num2str(itr), '-th iteration']);
    drawnow;
    
    %% all submatrices on the original permutation
    figid = figid + 1;
    figure(figid);
    
    z_all = ones(N1, N2, 3);
           
    % simple overwrite
    for k=1:K
        z_k = z1(:, k) * z2(:, k)';
        [i_idx, j_idx] = find(z_k > 0);
        for n=1:length(i_idx)
            ii = i_idx(n);
            jj = j_idx(n);
            z_all(ii, jj, :) = reshape(ccc(k,:), [1 1 3]);
         end % end n-for
    end % end k-for
    image(z_all);
    xlabel('i')
    ylabel('j')
    title(['all submatrices'])
    drawnow;
    
    %% sorted clustering on matrix
    
    if(SHOW_CLUSTER_FIGS)
        for k=1:K
            
            if DEBUG
                display('#');
                display(['# Show submatrix k= ', num2str(k)]);
                display('#');
            end
            
            % sort the item index
            z1k_idx = find( z1(:, k) == 1 )';
            z2k_idx = find( z2(:, k) == 1 )';
            
            z1k_diff_idx = setdiff(1:N1, z1k_idx);
            z2k_diff_idx = setdiff(1:N2, z2k_idx);
            
            sorted_item_idx1 = [z1k_idx z1k_diff_idx];
            sorted_item_idx2 = [z2k_idx z2k_diff_idx];
            
            if DEBUG
            display(['number of active objects in domain 1=', num2str(length(z1k_idx))])
            %z1k_idx'
            display(['number of active objects in domain 2=', num2str(length(z2k_idx))])
            %z2k_idx'
            end
            
            % sort the observation
            xx_sorted = plaid.xx;
            xx_sorted = xx_sorted(sorted_item_idx1, :);
            xx_sorted = xx_sorted(:, sorted_item_idx2);
            
            figid = figid + 1;
            figure(figid);
            imagesc(xx_sorted);
            colormap(1 - gray);
            
            xlabel('domain 1 i (sorted)');
            ylabel('domain 2 j (sorted)');
            
            % lines to separate the sub matrix
            
            %k_mod = mod(k,64) + 1;
            %col = ccc(k_mod, :);
            col = ccc(k, :);
            hline1 = line( [0 length(z2k_idx)], [length(z1k_idx) length(z1k_idx)], 'LineWidth', 2, 'Color', col);
            hline2 = line( [length(z2k_idx) length(z2k_idx)], [0 length(z1k_idx)], 'LineWidth', 2, 'Color', col);
            
            drawnow;
            
        end
    end
    
end
