%----
% Domain-adaptive deep network compression, ICCV 2017
% Code by Marc Masana
%----
%   acc=eval_acc(net,imdb) evaluatesthe accuracy of the given network on
%   the given imdb dataset for the test samples.
%
%   This function passes one image through the network at a time. It can
%   easily be modified to be used with a chosen batch size in order to
%   improve computation time. For that, it is recommended to read images
%   with vl_imreadjpeg function.
%
%   Parameters:
%       net: DagNN object containing the network to evaluate.
%       imdb: imdb struct for the given dataset.
%
function acc=eval_acc(net,imdb)

    % set up network
    inp_lays = net.getInputs();
    out_lays = net.getOutputs();
    net.conserveMemory = true;
    net.mode = 'test';
    net.move('gpu');
    % initialization
    indx=find(imdb.images.set==3); %test samples
    numImages=length(indx);
    scores=zeros(length(imdb.meta.classes),numImages);
    [H, W, ~] = size(imdb.images.data_mean);
    % evaluation
    for img = 1:numImages
        % obtain and preprocess an image
        im = imread(imdb.images.filenames{indx(img)});
        im_ = single(im);
        % if image is 2D, replicate for 3-channel
        if ndims(im_)==2
            im_ = repmat(im_,[1 1 3]);
        end
        im_ = single(imresize(im_,[H,W]));
        im_ = im_ - imdb.images.data_mean;
        % evaluate image
        net.eval({inp_lays{1}, gpuArray(im_)});
        % extract scores
        scores(:,img) = squeeze(gather(net.vars(net.getVarIndex(out_lays{1})).value));
    end
    % compute accuracy metric
    [~,pred_label]=max(scores,[],1);
    acc=100*sum(pred_label==imdb.images.labels(imdb.images.set==3))/size(pred_label,2);

end
