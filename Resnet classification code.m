clc;
localDir = "Enter the augmented image folder";

% Create image datastore
imageDir = fullfile(localDir, "Give the class lavel of the dataset");
imds = imageDatastore(imageDir, IncludeSubfolders=true, LabelSource="foldernames");

% Display a tile of random images
numImages = numel(imds.Labels);
idx = randperm(numImages, 16);
I = imtile(imds, 'Frames', idx);
figure
imshow(I);

classNames = categories(imds.Labels);
numClasses = numel(classNames);

[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, "give the validation and testing image size");

% Load a pretrained network
net = resnet50;
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);

newFcLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraph = replaceLayer(lgraph, 'fc1000', newFcLayer);

newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassLayer);

% Create augmented image datastores without data augmentation
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'ColorPreprocessing', 'gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

% Reset GPU before training
reset(gpuDevice(1));

% Set training options with a reduced mini-batch size and multi-GPU execution environment
options = trainingOptions("adam", ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 100, ... % Further reduce the mini-batch size
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 5, ...
    'Plots', "training-progress", ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'single-gpu', ...
    'CheckpointPath', tempdir, ...
    'Shuffle', 'every-epoch');

% Train the network using trainNetwork
net = trainNetwork(augimdsTrain, lgraph, options);

% Classify the validation images
[YPred, scores] = classify(net, augimdsValidation);

% Get the true labels
YValidation = imdsValidation.Labels;

% Compute the confusion matrix
confMat = confusionmat(YValidation, YPred);

% Plot the confusion matrix
figure
confusionchart(confMat, classNames);
title('Confusion Matrix for Validation Data');
