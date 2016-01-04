function [nn, DD, VV] = readUCIBoW(file_name)
% function [nn, DD, VV] = readUCIBoW(file_name)
%
% Read UCI-ML Bag-of-Words format data file. 
%
% Format looks like this: 
% <top of the file>
% D - number of docs
% W - number of vocabulary
% NNZ - number of nonzero lines
% docID wordID count - nonzero words in docs. 
% docID wordID count 
% ...
%
% input: 
% file_name     - string, the name (path) of the data file
%
% output: 
% nn            - DD by VV amtrix, the counts of word occurences. 
%                 nn(d,v) indicate the counts of word v in the d-th
%                 document. 
% DD            - scaler, number of documents. 
% VV            - scaler, size of vocabulary. 
%
% Written by Katsuhiko Ishiguro <ishiguro@cslab.kecl.ntt.co.jp>
% April 05, 2010. 

DEBUG = 0;

%% constants
% read the first three lines
fid = fopen(file_name);
A_1 = textscan(fid, '%f', 3);
fclose(fid);
A = A_1{1};
%A = textread(file_name, '%d', 3);

DD = A(1);
VV = A(2);
NNZ = A(3);
if(DEBUG)
    DD
    VV
    NNZ        
end

%% data
nn = zeros(DD, VV);

% read the data lines
fid = fopen(file_name);
datum = textscan(fid, '%f %f %f', 'headerlines', 3);
fclose(fid);
doc_index = datum{1};
voc_index = datum{2};
count_data = datum{3};
%[doc_index voc_index count_data] = textread(file_name, ...
%    '%d %d %d', 'headerlines', 3); 

% aggregate the data into counts
for i=1:length(doc_index)
    d = doc_index(i);
    v = voc_index(i);
    c = count_data(i);
    nn(d, v) =  c;
end % end i-for
    
 if(DEBUG)
    nonzero = find(nn ~= 0);
    length(nonzero)
    NNZ
    length(doc_index)     
 end