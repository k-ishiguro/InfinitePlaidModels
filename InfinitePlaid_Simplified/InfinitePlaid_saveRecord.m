function InfinitePlaid_saveRecord(plaid, recorder, save_name_prefix, itr, Z_file_name, SHOW_FIG, SAVE_FIG)
% InfinitePlaid_saveRecord(plaid, recorder, save_name_prefix, itr)
%
% save record histories in text file for Infinite Plaid Model.
%
% plaid             - MATLAB class instance, InfinitePlaid model.
% recorder          - recorder instance.
% save_name_prefix  - string, prefix for files
% itr               - scaler, number of current interation.
% Z_file_name       - String, ground truth of clustering assignments.
% SHOW_FIG          - flag, if show figures, set 1.
% SAVE_FIG          - flag, if save figures, set 1.
%
% Written by Katsuhiko Ishiguro <ishiguro.katsuhiko@lab.ntt.co.jp>
% Last update: 18/11/15 (dd/mm/yy)

matname = [save_name_prefix, '_record_itr', num2str(itr), '.mat']
eval(['save ', matname, ' recorder']);

filename = [save_name_prefix, '_record_itr', num2str(itr), '.dat']
fid = fopen(filename, 'w');

K = plaid.KK;
N1 = plaid.NN1;
N2 = plaid.NN2;

loglk = recorder.loglk;
loglk_marginal = recorder.loglk_marginal;
test_loglk = recorder.test_loglk;
test_loglk_marginal = recorder.test_loglk_marginal;
if ~strcmp(Z_file_name, 'null')
        nmi_history = recorder.nmi_history;
end
KK_history = recorder.KK_history;

%% write several constants
fprintf(fid, ['NN1=', num2str(N1), '\n']);
fprintf(fid, ['NN2=', num2str(N2), '\n']);
fprintf(fid, ['KK=', num2str(K), '\n']);
%fprintf(fid, ['TT=', num2str(TT), '\n']);

%% summary of the statistics, in .dat
fprintf(fid, ['training-data log likelihood=', num2str(loglk(itr)), '\n']);
fprintf(fid, ['training-data log likelihood(marginal)=', num2str(loglk_marginal(itr)), '\n']);
fprintf(fid, ['test-data log likelihood=', num2str(test_loglk(itr)), '\n']);
fprintf(fid, ['test-data log likelihood(marginal)=', num2str(test_loglk_marginal(itr)), '\n']);
if ~strcmp(Z_file_name, 'null')
       fprintf(fid, ['NMI=', num2str(nmi_history(itr)), '\n']);
end
fprintf(fid, ['number of sub-matrices=', num2str(KK_history(itr)), '\n']);
fprintf(fid, ['number of sub-matrices(marginal)', num2str( mean(KK_history(1:itr)) ), '\n']);

%% save histories

% save the history of log likelihood
fprintf(fid, 'training data log likelihood: \n');
for i=1:itr
    fprintf(fid, [num2str(i), ' ', num2str(loglk(i)), '\n']);
end

% save the history of log likelihood (marginalized)
fprintf(fid, 'training data log likelihood (marginalized): \n');
for i=1:itr
    fprintf(fid, [num2str(i), ' ', num2str(loglk_marginal(i)), '\n']);
end

% save the history of log likelihood
fprintf(fid, 'test data log likelihood: \n');
for i=1:itr
    fprintf(fid, [num2str(i), ' ', num2str(test_loglk(i)), '\n']);
end

% save the history of log likelihood (marginalized)
fprintf(fid, 'test data log likelihood (marginalized): \n');
for i=1:itr
    fprintf(fid, [num2str(i), ' ', num2str(test_loglk_marginal(i)), '\n']);
end

if ~strcmp(Z_file_name, 'null')
    % save the history of NMI
    fprintf(fid, 'NMI: \n');
    for i=1:itr
        fprintf(fid, [num2str(i), ' ', num2str(nmi_history(i)), '\n']);
    end
end

% save the history of instantiated sub-matrices
fprintf(fid, 'history of K instantiated sub-matrices: \n');
for i=1:itr
    fprintf(fid, [num2str(i), ' ', num2str(KK_history(i)), '\n']);
end

fclose(fid);

%% figures
if SHOW_FIG | SAVE_FIG
    
    figid = 120;
    figid = figid + 1;
    f = figure(figid);
    set(f, 'visible', 'off');
    plot(1:itr, loglk(1:itr));
    xlabel('iterations');
    title('Averaged Training data Log Likelihood');
    
    if SHOW_FIG
        set(f, 'visible', 'on');
        drawnow;
    end
    
    if SAVE_FIG
        eval(['figname = ''', save_name_prefix, '_LogLk', ''';']);
        eval(['figname_fig = ''', figname, '.fig'';']); saveas(gcf,figname_fig);
        eval(['figname_eps = ''', figname, '.eps'';']); saveas(gcf,figname_eps,'epsc');
        eval(['figname_png = ''', figname, '.png'';']); saveas(gcf,figname_png);
    end
    
    figid = figid + 1;
    f = figure(figid);
    set(f, 'visible', 'off');
    plot(1:itr, test_loglk(1:itr));
    xlabel('iterations');
    title('Averaged Test data Log Likelihood');
    
    if SHOW_FIG
        set(f, 'visible', 'on');
        drawnow;
    end
    if SAVE_FIG
        eval(['figname = ''', save_name_prefix, '_TestLogLk', ''';']);
        eval(['figname_fig = ''', figname, '.fig'';']); saveas(gcf,figname_fig);
        eval(['figname_eps = ''', figname, '.eps'';']); saveas(gcf,figname_eps,'epsc');
        eval(['figname_png = ''', figname, '.png'';']); saveas(gcf,figname_png);
    end
    
    figid = figid + 1;
    f = figure(figid);
    set(f, 'visible', 'off');
    plot(1:itr, loglk_marginal(1:itr));
    xlabel('iterations');
    title('Averaged Training data Marginal Log Likelihood');
    
    if SHOW_FIG
        set(f, 'visible', 'on');
        drawnow;
    end
    
    if SAVE_FIG
        eval(['figname = ''', save_name_prefix, '_MarginalLogLk', ''';']);
        eval(['figname_fig = ''', figname, '.fig'';']); saveas(gcf,figname_fig);
        eval(['figname_eps = ''', figname, '.eps'';']); saveas(gcf,figname_eps,'epsc');
        eval(['figname_png = ''', figname, '.png'';']); saveas(gcf,figname_png);
    end
    
    figid = figid + 1;
    f = figure(figid);
    set(f, 'visible', 'off');
    plot(1:itr, test_loglk_marginal(1:itr));
    xlabel('iterations');
    title('Averaged Test data Marginal Log Likelihood');
    
    if SHOW_FIG
        set(f, 'visible', 'on');
        drawnow;
    end
    
    if SAVE_FIG
        eval(['figname = ''', save_name_prefix, '_MarginalTestLogLk', ''';']);
        eval(['figname_fig = ''', figname, '.fig'';']); saveas(gcf,figname_fig);
        eval(['figname_eps = ''', figname, '.eps'';']); saveas(gcf,figname_eps,'epsc');
        eval(['figname_png = ''', figname, '.png'';']); saveas(gcf,figname_png);
    end
    
    if ~strcmp(Z_file_name, 'null')
               
        figid = figid + 1;
        f = figure(figid);
        set(f, 'visible', 'off');
        plot(1:itr, nmi_history(1:itr));
        xlabel('iterations');
        title('NMI');
        
        if SHOW_FIG
            set(f, 'visible', 'on');
            drawnow;
        end
        
        if SAVE_FIG
            eval(['figname = ''', save_name_prefix, '_NMI', ''';']);
            eval(['figname_fig = ''', figname, '.fig'';']); saveas(gcf,figname_fig);
            eval(['figname_eps = ''', figname, '.eps'';']); saveas(gcf,figname_eps,'epsc');
            eval(['figname_png = ''', figname, '.png'';']); saveas(gcf,figname_png);
        end
    end
    
    figid = figid + 1;
    f = figure(figid);
    set(f, 'visible', 'off');
    plot(1:itr, KK_history(1:itr));
    xlabel('iterations');
    title('Instantiated sub-matrices');
    
    if SHOW_FIG
        set(f, 'visible', 'on');
    end
    
    if SAVE_FIG
        eval(['figname = ''', save_name_prefix, '_Khistory', ''';']);
        eval(['figname_fig = ''', figname, '.fig'';']); saveas(gcf,figname_fig);
        eval(['figname_eps = ''', figname, '.eps'';']); saveas(gcf,figname_eps,'epsc');
        eval(['figname_png = ''', figname, '.png'';']); saveas(gcf,figname_png);
    end
    
    
end

