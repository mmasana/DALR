%----
% Domain-adaptive deep network compression, ICCV 2017
% Code by Marc Masana
%----
%   [net] = compressLayerDALR(net,layer,acts,t,lambda) modifies the given
%   network by applying the DALR method to the chosen layer. That layer
%   will be substituted by two new layers which correspond to the A and
%   B matrices from the method.
%
%   Parameters:
%       net: DagNN object containing the network to compress.
%       layer: string with the name of the layer to compress.
%       acts: activations of the chosen layer for the given network.
%       t: number of dimensions to keep from the kernel.
%       lambda: value for the regularization term.
%
function [net] =  compressLayerDALR(net,layer,acts,t,lambda)

    % get layer parameter variables
    layParamsName = net.layers(net.getLayerIndex(layer)).params;
    % extract those parameters
    layerWeight = net.params(net.getParamIndex(layParamsName{1})).value;
    layerBiases = net.params(net.getParamIndex(layParamsName{2})).value;
    % reshape weight matrix
    num1 = size(layerWeight,1);
    num2 = size(layerWeight,2);
    num3 = size(layerWeight,3);
    layerWeight = squeeze(reshape(layerWeight,1,1,num1*num2*num3,size(layerWeight,4)));
    % apply DALR - Domain Adaptive Low Rank matrix decomposition
    [U2,~,~]=svd(layerWeight'*acts);
    XXT=acts*acts';
    t=ceil(t*size(layerWeight,2));
    A=U2(:,1:t);
    B=U2(:,1:t)'*layerWeight'*XXT*inv(XXT+lambda*eye(size(acts,1)));
    % reshape matrices back
    A = reshape(A',1,1,size(A,2),size(A,1));
    B = reshape(B',num1,num2,num3,size(B,1));
    % add two new layers
    inputs = char(net.layers(net.getLayerIndex(layer)).inputs);
    outputs = char(net.layers(net.getLayerIndex(layer)).outputs);
    net.addLayer(strcat(layer,'_A'), dagnn.Conv('size', [1 1 size(layerWeight,3) t], 'hasBias', false, 'stride', [1, 1], 'pad', [0 0 0 0]), {inputs}, {strcat(layer,'_aux')},  {strcat(layer,'_Af')});
    net.addLayer(strcat(layer,'_B'), dagnn.Conv('size', [1 1 t size(layerWeight,4)], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {strcat(layer,'_aux')}, {outputs},  {strcat(layer,'_Bf')  strcat(layer,'_Bb')});
    % add weights for first added layer
    layParamsName = net.layers(net.getLayerIndex(strcat(layer,'_A'))).params;
    net.params(net.getParamIndex(layParamsName{1})).value = B;
    % add weights for second added layer
    layParamsName = net.layers(net.getLayerIndex(strcat(layer,'_B'))).params;
    net.params(net.getParamIndex(layParamsName{1})).value = A;
    net.params(net.getParamIndex(layParamsName{2})).value = layerBiases;
    % wrap and remove old layer
    net.removeLayer(layer);

end
