% dataDir should be the path to the folder for the CUB_200_2011 dataset
function imdb = createIMDB_Birds(dataDir)
% imdb is a matlab struct with several fields, such as:
%	- images: contains data, labels, ids dataset mean, etc.
%	- meta: contains meta info useful for statistics and visualization
%	- any other you want to add

    imdb = struct();

    % we can initialize part of the structures already
    classes = strread(num2str(1:200),'%s')';
    meta.sets = {'train','val','test'};
    meta.classes = classes;
    meta.rangemultiplier=1;

    % VGG network average image
    averageImage(1,1,1) = 123.680;
    averageImage(1,1,2) = 116.779;
    averageImage(1,1,3) = 103.939;
    % this will contain the mean of the training set
    images.data_mean = repmat(averageImage,224,224);

    % we do not save the images, they will be read during execution
    images.data = [];

    % a label per image
    [~,labels] = textread(strcat(dataDir,'/image_class_labels.txt'),'%d %d');
    images.labels = single(labels)';
    % set splits - labels we use train=1, val=2, test=3
    [~,splits] = textread(strcat(dataDir,'/train_test_split.txt'),'%d %d');
    images.set = uint8(zeros(1, size(splits,1)));
    % divide trainval into train and test
    images.set(find(splits==1)) = uint8(1); % NO VALIDATION
    images.set(find(splits==0)) = uint8(3);
    % image names
    [~,image_names] = textread(strcat(dataDir,'/images.txt'),'%d %s');
    for m=1:size(splits,1)
        images.filenames{m} = sprintf('%s/images/%s',dataDir,image_names{m});
    end

    imdb.meta = meta;
    imdb.images = images;

end
