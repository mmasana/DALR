%----
% Domain-adaptive deep network compression, ICCV 2017
% Code by Marc Masana
%----
%   This script shows a basic example on how to use the DALR method (Domain
%   Adaptive Low Rank matrix decomposition). A vanilla fine-tuned network
%   based on VGG-19 for CUB_200_2011 Birds dataset is used. The activations
%   for the fc7 layer are extracted and then used to compress the network
%   on different compression rates using the DALR method. For more details
%   on the method, please check the corresponding paper.
%
%% Run the code
% use matconvnet stuff
run <path_to_matconvnet>/matlab/vl_setupnn
% load imdb
imdb = createIMDB_Birds(path_to_dataset);
% get activations
load('../../nets/birds_vgg19_net.mat');
[acts] = getActivations(net,imdb,'fc7','../../data/CUB_200_Birds/VGG19_fc7_acts.mat');
% original accuracy
net = dagnn.DagNN.loadobj(net);
acc = eval_acc(net,imdb);
disp(['Original accuracy: ', num2str(acc,'%2.4f')]);
% apply compression
lambda=1000;
compress = [32 64 128 256 512 1024 4096] / 4096.0;
acc = zeros(1,size(compress,2));
for point=size(compress,2):-1:1
    % load network
    net = load('../../nets/birds_vgg19_net.mat');
    net = dagnn.DagNN.loadobj(net.net);
    % compress network
    newNet =  compressLayerDALR(net,'fc7',acts,compress(point),lambda);
    % test
    acc(1,point) = eval_acc(newNet,imdb);
    disp(['Accuracy when keep ',num2str(100*compress(1,point),'%2.2f'),'% dim: ', num2str(acc(1,point),'%2.4f')]);
end
plot(acc);
