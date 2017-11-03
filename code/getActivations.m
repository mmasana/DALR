%----
% Domain-adaptive deep network compression, ICCV 2017
% Code by Marc Masana
%----
%   [acts] = getActivations(net,imdb,layer,fileName) extracts the
%   activations from the chosen layer of the given network for the non-test
%   samples of the imdb dataset. Activations are extracted or loaded
%   depending on if the filename already exists.
%
%   This function passes one image through the network at a time. It can
%   easily be modified to be used with a chosen batch size in order to
%   improve computation time. For that, it is recommended to read images
%   with vl_imreadjpeg function.
%
%   Parameters:
%       net: DagNN object containing the network to extract features from.
%       imdb: imdb struct for the given dataset.
%       layer: string with the name of the layer to extract features from.
%       fileName: if the filename exists, it loads the activations,
%                 otherwise it extracts the features and saves them on that
%                 location.
%
function [acts] = getActivations(net,imdb,layer,fileName)
    
    % check if filename exists, if so, load the activations
    if exist(fileName, 'file') == 2
        load(fileName);
    else
        % get layers info
        net = dagnn.DagNN.loadobj(net);
        inp_lays = net.getInputs();
        layParamsName = net.layers(net.getLayerIndex(layer)).params;
        layVarsName = net.layers(net.getLayerIndex(layer)).inputs;
        dims = size(net.params(net.getParamIndex(layParamsName{1})).value);
        % get imdb info
        indx=find(imdb.images.set~=3); %non-test samples
        numImages = length(indx);
        [H, W, ~] = size(imdb.images.data_mean);
        % allocate memory
        acts = zeros(prod(dims(1:end-1)),numImages);
        % prepare network
        net = dagnn.DagNN.loadobj(net);
        net.conserveMemory = false;
        net.mode = 'test';
        net.move('gpu');
        % extract features
        for img = 1:numImages
            % obtain and preprocess an image
            im = imread(imdb.images.filenames{indx(img)});
            im_ = single(im);
            % if image is 2D, replicate for 3-channel
            if ndims(im_)==2
                im_ = repmat(im_,[1 1 3]);
            end
            im_ = imresize(im_,[H,W]);
            im_ = im_ - imdb.images.data_mean;
            % run the CNN
            net.eval({inp_lays{1}, gpuArray(im_)});
            % store the features
            acts(:,img) = squeeze(gather(net.vars(net.getVarIndex(layVarsName)).value));
        end
        % save file
        save(fileName,'acts','-v7.3');
    end
end
