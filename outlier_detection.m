function [A_new,A_outs,F] = outlier_detection(A_in)
% function [A_new,A_outs] = outlier_detection(A)
% 
% This function implements outlier detection with methods shown in
% Bradley Hobbs and Panagiotis Artemiadis, “A Systematic Method for Outlier Detection in Human Gait Data,” in the 2022 IEEE 17th International Conference on Rehabilitation Robotics (ICORR), 2022
%
% Input: A - matrix of values with gait cycle periods in each row and with 
%            each gait cycle sample observation in each column, OR
%            a cell array of arbitrary size, with each cell consisting of a
%            matrix of values with gait cycle periods in each row and with
%            each gait cycle sample observation in each column
% 
% Output: A_new - Matrix or cell array in the same form as input but row
%                 reduced to only contain regular, non-outlier gait cycles
%        A_outs - Matrix or cell array in the same form as input but row
%                 reduced to only contain only outlier gait cycles.
%
% If A contains X unique gait cycles, A_new and A_outs combined will
% contain all X unique gait cycles between them

%% Prepare input
if iscell(A_in)  
    disp('inside') % if A is a cell with multiple sources   
    if ~(numel(unique(cellfun('size',A_in,1)))==1)                          % if number of periods for all cells in A are NOT equivalent 
        error('Ensure number of periods are equivalent for all cells in A') % Throw error if the number of periods for all sources in A are not equal
    end                                                                     % end
    [nA,mA] = size(A_in);                                                   % Store input dimensions for cell shaping and reshaping
    A       = reshape(A_in,1,nA*mA);                                        % Convert cell array A into cell array A with size {1,nA*mA}
    fsA     = cellfun('size',A,2);                                          % Store number of samples for each source for cell reshaping
    ns      = size(A,2);                                                    % number of total sensor sources
else                                                                        % else A is a matrix from a single source
    ns = 1;                                                                 % single source A
    A{1} = A_in;                                                            % Convert to single cell
end                                                                         % end

%% Run flagging procedures
h = cell(1,ns);                                % Prepare empty cell
for i = 1:ns                                   % for each source
    h{1,i} =     shape_based_procedure(A{i})'; % Run shape-based procedure
    h{2,i} =   feature_based_procedure(A{i})'; % Run feature-based procedure
    h{3,i} =      time_based_procedure(A{i})'; % Run time-based procedure
    h{4,i} = amplitude_based_procedure(A{i})'; % Run amplitude-based procedure
    h{5,i} =     stats_based_procedure(A{i})'; % Run statistics-based procedure
end                                            % end
H     = cell2mat(reshape(h,1,numel(h)));       % Convert cell into matrix of size [1xnf*ns]
A_mat = cell2mat(A);                           % Convert cell array into matrix
    
%% Calculate new data
[F,confidence,cutoff] = pfdp(H,A_mat);                           % Find final outlier cycles
A_new  = A_mat(~F,:);                                            % Extract new data without outliers
A_outs = A_mat(F,:);                                             % Extract outliers only
if ns ~= 1                                                       % If input was not originally a matrix                                    
    A_new = reshape(mat2cell(A_new,size(A_new,1),fsA),nA,mA);    % Shape new data into the same cell dimensions as A
    A_outs = reshape(mat2cell(A_outs,size(A_outs,1),fsA),nA,mA); % Shape outliers into the same cell dimensions as A
end

return % no plotting 

%% Plot results
if ns == 1 
    % Plot gait cycles in time domain
    figure; set(gcf,'color','w'); hold on
    title 'Color-Coded Input Data'
    xlabel('Percent Gait Cycle (L-HS to L-HS)','fontsize',14)
    ylabel('Artificial VSM Activity','fontsize',14)
    p1 = plot(A_new','g');
    p2 = plot(A_outs','r');
    legend([p1(1) p2(1)],'New Data','Outliers','Location','best')
    
    % Plot confidence for each period against chosen cutoff individually
    figure; set(gcf,'color','w'); hold on
    title 'Confidence For Each Gait Cycle'
    xlabel('Gait Cycle Number','fontsize',14)
    ylabel('Confidence Level','fontsize',14)
    plot(confidence,'o')
    plot([0 length(confidence)],[cutoff cutoff],'r')
    
    % Plot confidence for each period against chosen cutoff (as summation)
    figure; set(gcf,'color','w'); hold on
    title 'Confidence For Each Gait Cycle As Summation'
    xlabel('Gait Cycle Number','fontsize',14)
    ylabel('Confidence Level','fontsize',14)
    all = ones(length(confidence),1).*sort(unique(confidence)','descend');
    confidence_sum = all.*(all<=confidence);
    plot(confidence_sum,'o');
    plot([0 length(confidence)],[cutoff cutoff],'r')
end
end

function [H] = shape_based_procedure(A)
% function [H] = shape_based_procedure(A)
% 
% This function implements piecewise variant of Median Absolute Deviation (pMAD) 
% to flag gait cycles as potential outliers from the time domain 
% and based on acute shapes of the waveform
%
% Input: A - matrix of data from a single source
%          - has nc rows (gait cycles) and np columns (samples)
% 
% Output: H - Logical vector of flags, with one flag for each gait cycle
%           - has nc rows (gait cycles) and 1 column

np = size(A,2);             % np number of samples per gait cycle   
ML = isoutlier(A,'median'); % MAD implementation for each sample
M  = sum(ML,2);             % sum each column
t1 = 0.04;                  % percentage threshold desired
H  = (M > t1*np)';          % flag gait cycle if above percentage allowable
end

function [H] = feature_based_procedure(A)
% function [H] = feature_based_procedure(A)
% 
% This function implements Principal Component Analysis (PCA) 
% to flag gait cycles as potential outliers from the Principal Component (PC) domain
% and based on prominent features of the waveform
%
% Input: A - matrix of data from a single source
%          - has nc rows (gait cycles) and np columns (samples)
% 
% Output: H - Logical vector of flags, with one flag for each gait cycle
%           - has nc rows (gait cycles) and 1 column

[~,WL,~,~,~] = pca(A);             % all PC scores
h1  = isoutlier(WL(:,1),'median'); % MAD implementation for first PC
h2  = isoutlier(WL(:,2),'median'); % MAD implementation for second PC
H   = any(h1+h2,2)';               % Only one sub-flag necessary for final flag

return % no plotting

WL1 = WL(:,1);                     % Ppc_alls of PC1
WL2 = WL(:,2);                     % Ppc_alls of PC2

H1found = find(h1==1);
H2found = find(h2==1);

figure; set(gcf,'color','w'); hold on
scatter(WL1,WL2,'g','LineWidth',2);
box on
set(gca,'FontSize',14);
title('Feature-Based Procedure Results','FontSize',14);
xlabel('PC1 Ppc_alls','FontSize',14)
ylabel('PC2 Ppc_alls','FontSize',14)
p1 = scatter(WL(:,1),WL(:,2),'g','LineWidth',2);
p2 = scatter(WL(H1found,1),WL(H1found,2),'r','LineWidth',2);
     scatter(WL(H2found,1),WL(H2found,2),'r','LineWidth',2);
p4 = scatter(WL(1:40,1),WL(1:40,2),'ko','LineWidth',1);
legend([p1(1),p2(1),p4(1)],'Not Flagged','Flagged','Outliers')
end

function [H] = time_based_procedure(A)
% function [H] = time_based_procedure(A)
% 
% This function implements discrete Fast Fourier Transform (FFT) 
% to flag gait cycles as potential outliers from the frequency domain
% and based on the time differences in the waveform
%
% Input: A - matrix of data from a single source
%          - has nc rows (gait cycles) and np columns (samples)
% 
% Output: H - Logical vector of flags, with one flag for each gait cycle
%           - has nc rows (gait cycles) and 1 column

np           = size(A,2);                    % np number of samples per gait cycle
YL           = fft(A')';                     % FFT result
spectrum2    = abs(YL/np);                   % double-sided spectrum
Y            = spectrum2(:,1:np/2+1);        % single-sided spectrum
Y(:,2:end-1) = 2*Y(:,2:end-1);               % single-sided spectrum
h            = isoutlier(Y(:,1:4),'median'); % MAD implementation for first 4Hz fo spectrum
H            = any(h,2)';                    % flag gait cycle if any frequency shows abnormality

return % no plotting

figure; set(gcf,'color','w'); hold on
title('Time-Based Procedure Results','FontSize',14);
xlabel('Frequency [Hz]','FontSize',14)
ylabel('Amplitude','FontSize',14)
p1 = plot(Y(1:40,:)','r');              % plot artificial outliers
p2 = plot(Y(41:end,:)','g');            % plot artificial normal gait cycles
p3 = plot(Y(H,:)','k','linewidth',0.5); % plot flagged gait cycles
legend([p1(1),p2(1),p3(1)],'Outliers','Normal','Flagged')
end

function [H] = amplitude_based_procedure(A)
% function [H] = amplitude_based_procedure(A)
% 
% This function implements discrete Fast Fourier Transform (FFT) 
% to flag gait cycles as potential outliers from 1 dimensional integral scores
% and based on the amplitude differences in the waveform
%
% Input: A - matrix of data from a single source
%          - has nc rows (gait cycles) and np columns (samples)
% 
% Output: H - Logical vector of flags, with one flag for each gait cycle
%           - has nc rows (gait cycles) and 1 column

I = trapz(A,2);             % calculate integral along periods (rows)
H = isoutlier(I,'median')'; % MAD implementation for vector of integral scores

return % no plotting

figure; set(gcf,'color','w'); hold on
set(gca,'FontSize',14);
title('Amplitude-Based Procedure Results','FontSize',14);
xlabel('Integral Scores','FontSize',14)
ylabel('Integral Scores','FontSize',14)
p1 = scatter(I(H==0),I(H==0),'go','LineWidth',2);
p2 = scatter(I(H==1),I(H==1),'ro','LineWidth',2);
p3 = scatter(I(1:40),I(1:40),'ko','Linewidth',1);
legend([p1(1),p2(1),p3(1)],'Not Flagged','Flagged','Outliers','location','northwest')
end

function [H] = stats_based_procedure(A)
% function [H] = stats_based_procedure(A)
% 
% This function implements Generalized Extreme Studentized Deviate (GESD) 
% to flag gait cycles as potential outliers from raw gait cycle input
%
% Input: A - matrix of data from a single source
%          - has nc rows (gait cycles) and np columns (samples)
% 
% Output: H - Logical vector of flags, with one flag for each gait cycle
%           - has nc rows (gait cycles) and 1 column

[~,G] = rmoutliers(A,'gesd'); % Grubbs-based GESD test
H     = G';                   % Transpose for consistency 
end

function [F,conf,C_conf] = pfdp(H,A)
% function [F_final,conf,C_conf] = pfdp(H,A)
% 
% This function implements a Post-Flagging Decision Procedure (PFDP)
% based on statistical reasoning to make the final decision for 
% choosing outliers from all sources and all flagging procedures
%
% Input: A - matrix of data from all sources concatenated
%          - has nc rows (gait cycles) and np*ns columns (samples)
%
%        H - matrix of data from all binary flagging vectors and all sources concatenated
%          - has nc rows (gait cycles) and 5*ns columns (samples)
% 
% Output: F_final - Logical vector of flags, with one flag for each gait cycle
%                 - has nc rows (gait cycles) and 1 column
%
%            conf - vector of data containing the confidence that each 
%                   corresponding gait cycle is an outlier
%                 - has nc rows (gait cycles) and 1 column
%
%          C_conf - scalar value signifying the chosen cutoff value
%                   expressed as a confidence percentage

%% Calculate All Possible Combinations of Outliers based on confidence level        
ns5   = size(H,2);                                                          % Number of sources * number of flagging procedures
H     = sum(H,2);                                                           % Add up all flags for each gait cycle
conf  = H./ns5;                                                             % Calculate the confidence that each gait cycle is an outlier
eta   = zeros([1 ns5]);                                                     % Initialize vector of hypothesis results sums
method = 2;                                                                 % Method 1: statistics based approach used in paper. Method 2: faster approach
if method == 1                                                              % if method 1 is being used
    for C = 1:ns5                                                           % for all possible cutoff values C
        N      = A(H < C,:);                                                % Testing as tentative non-outliers N
        O      = A(H >= C,:);                                               % Testing as tentative outliers O
        etaXi  = ttest2(N,O,'alpha',.05,'tail','both','vartype','unequal'); % Use 2-sample t-test with unequal means at a 5% confidence interval for each column/vertical slice between the two sets
        eta(C) = sum(etaXi);                                                % Sum hypothesis testing results for all np*ns partitions
    end                                                                     % end

    %% Choose Maximum Value (Or Similar) of Hypothesis Testing Results
    [~,locs] = findpeaks(eta);        % Find local maximum values
    if isempty(locs)                  % if there are no local maximum values (usually due to too few sources)
        C_final = floor(ns5/2);       % Choose median value
    elseif ns5 <= 5                   % else in the case of at least one local maximum, and in the case of small number of sources and/or flagging procedures
        C_final = locs(1);            % Just choose the first local maximum
    else                              % else for large enough datasets, eta creates a curve similar to a concave down and decreasing exponential decay function 
        [~,C_final] = max(eta);       % Choose the local maximum nearest to the knee/elbow/bend of eta for the most conservative C_final.
    end                               % end
elseif method == 2                    % else if method 2 is being used
    C_final = median(H) + (3*std(H)); % non statistics-based alternative that requires less tuning
end                                   % end either method
C_conf = C_final/ns5;                 % calculate confidence percentage for each gait cycle
F      = (H >= C_final);              % Use C to determine final vector of outlier labels where periods of C_final or higher are outliers
end                       







