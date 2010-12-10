function features = Gist_Feature()

% This code compute features with diffrent parameters
% and save them in output files

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
tic;
HOMEIMAGES = 'dataset/tmp'
imageSize = 256; 
orientationsPerScale = [2 3 4 5; 3 3 3 3; 8 8 8 8; 2 2 2 2];
numberBlocks = [5 3 4 2];
fc_prefilt = [2 5 3 4];
test_number = size(numberBlocks,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute global features
scenes = dir(fullfile(HOMEIMAGES, '*.jpg'));
scenes = {scenes(:).name};
Nscenes = length(scenes);


features = cell(1,test_number);

% Compute features with diffrent parameters
for i = 1:test_number
    % Precompute filter transfert functions (only need to do this once, unless
    % image size is changes):
    G = createGabor(orientationsPerScale(i,:), imageSize);
    Nfeatures = size(G,3)*numberBlocks(i)^2;
    
    % Loop: Compute global features for all scenes
    F = zeros([Nscenes Nfeatures]);
    for n = 1:Nscenes
        disp([n Nscenes])
        img = imread(fullfile(HOMEIMAGES, scenes{n}));
        img = mean(img,3);
        if size(img,1) ~= imageSize
            img = imresize(img, [imageSize imageSize], 'bilinear');
        end
    
        output = prefilt(img, fc_prefilt(i));
        g = gistGabor(output, numberBlocks(i), G);
        F(n,:) = g;
    end
    a
    features{i} = F;
end

%save features
for i=1:length(features)
    tmp = features{i};
    save(['data/feat',int2str(i), '.txt'], 'tmp', '-ASCII');
end

%save param
save(['data/param.txt'], 'test_number', 'numberBlocks', 'fc_prefilt', 'orientationsPerScale', '-ASCII');


endmg,1) ~= imageSize
            img = imresize(img, [imageSize imageSize], 'bilinear');
        end
    
        output = prefilt(img, fc_prefilt(i));
        g = gistGabor(output, numberBlocks(i), G);
        F(n,:) = g;
    end
    a
    features{i} = F;
end

%save features
for i=1:length(features)
    tmp = features{i};
    save(['data/feat',int2str(i), '.txt'], 'tmp', '-ASCII');
end

%