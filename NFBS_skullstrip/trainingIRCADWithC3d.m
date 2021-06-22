%% 3-D Blood Vessel Extraction from MRI
% Train and cross validate a 3-D densenet

% Clear workspace
clearvars; close all; clc;

gpuDevice(1)

%Input filename and pathways
vName = 'inputIRCAD.json'; %load nifti data from inputIRCAD.csv 
jsonData = jsondecode(fileread(vName)); %changes the original filename 
dest = "IRCADwithC3d/liver_30epoch_7/";
desfold = pwd + "/IRCADwithC3d/liver_30epoch_7/";
epnum = 30;
conv = 16;

% Read file pathways into table
T = readtable(jsonData.fullFileName, 'Delimiter', jsonData.delimiter);
A = table2array(T);

volLoc = A(:, jsonData.volCol);
lblLoc = A(:, jsonData.lblCol);
idLoc = A(:, jsonData.idCol);

stoFoldername = jsonData.stoFoldername;
destination = fullfile(stoFoldername);

%% create datastores for processed labels and images
% Images Datapath % define reader
procVolReader = @(x) niftiread(x);
procVolLoc = fullfile(destination,'patient_CT');
procVolDs = imageDatastore(procVolLoc, ...
    'FileExtensions','.nii','LabelSource','foldernames','ReadFcn',procVolReader);

procLblReader =  @(x) uint8(niftiread(x));
procLblLoc = fullfile(destination,'vessel_lbl');
classNames = ["background","Vessel"];
pixelLabelID = [0 1];
procLblDs = pixelLabelDatastore(procLblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.nii','ReadFcn',procLblReader);

%kfold partition
num_images = length(procVolDs.Labels); %number of obervations for kfold
c1 = cvpartition(num_images,'kfold',20); %try with 4 instead of 5 
err = zeros(c1.NumTestSets,1);

C = cell(1, 20);
[idxTest] = deal(C);

for idxFold = 1:c1.NumTestSets
    idxTest{idxFold} = test(c1,idxFold); %logical indices for test set
    save('idxTest.mat','idxTest');
    save(desfold + '/idxTest.mat','idxTest');
    
    idxHold = training(c1,idxFold); %logical indices for training set-holdout partition
    imdsHold = subset(procVolDs,idxHold); %imds for holdout partition
    pxdsHold = subset(procLblDs,idxHold); %imds for holdout partition

    num_imdsHold = length(imdsHold.Labels); %number of obervations for holdout partition
    c2 = cvpartition(num_imdsHold,'holdout',0.25);
    
    idxVal = test(c2); %logical indices for val set
    imdsVal = subset(imdsHold,idxVal); %val imagedatastore
    pxdsVal = subset(pxdsHold,idxVal); %val pixelimagedatastore
   
    idxTrain = training(c2); %logical indices for training set
    imdsTrain = subset(imdsHold,idxTrain); %training imagedatastore
    pxdsTrain = subset(pxdsHold,idxTrain); %training pixelimagedatastore

%Need Random Patch Extraction on training and validation Data
    patchSize = [64 64 64];
    patchPerImage = 16;
    miniBatchSize = 4; %originally was 8

%training patch datastore
    trPatchDs = randomPatchExtractionDatastore(imdsTrain,pxdsTrain,patchSize, ...
    'PatchesPerImage',patchPerImage);
    trPatchDs.MiniBatchSize = miniBatchSize;

%validation patch datastore
    valPatchDs = randomPatchExtractionDatastore(imdsVal,pxdsVal,patchSize, ...
    'PatchesPerImage',patchPerImage);
    valPatchDs.MiniBatchSize = miniBatchSize;

%% Create Layer Graph
% Create the layer graph variable to contain the network layers.
%define n as number of channels
n_layer = 1;
lgraph = layerGraph();

%% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.
% Helper function for densenet3d upsample3dLayer.m

tempLayers = [
    image3dInputLayer([64 64 64 n_layer],"Name","input","Normalization","none")
    batchNormalizationLayer("Name","BN_Module1_Level1")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module1_Level2")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,2,"Name","concat_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","BN_Module2_Level1")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module2_Level2")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,2,"Name","concat_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","BN_Module3_Level1")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module3_Level2")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,2,"Name","concat_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","BN_Module4_Level1")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module4_Level2")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat_4")
    upsample3dLayer([2 2 2],conv,"Name","upsample_Module4","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat3")
    batchNormalizationLayer("Name","BN_Module5_Level1")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module5_Level2")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat_5")
    upsample3dLayer([2 2 2],conv,"Name","upsample_Module5","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat2")
    batchNormalizationLayer("Name","BN_Module6_Level1")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module6_Level2")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat_6")
    upsample3dLayer([2 2 2],conv,"Name","upsample_Module6","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat1")
    batchNormalizationLayer("Name","BN_Module7_Level1")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module7_Level1","Padding","same")
    reluLayer("Name","relu_Module7_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module7_Level2")
    convolution3dLayer([3 3 3],conv,"Name","conv_Module7_Level2","Padding","same")
    reluLayer("Name","relu_Module7_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat")
    batchNormalizationLayer("Name","BN_Module7_Level3")
    convolution3dLayer([1 1 1],2,"Name","ConvLast_Module7")
    softmaxLayer("Name","softmax")
    dicePixelClassification3dLayer("output")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

lgraph = connectLayers(lgraph,"relu_Module1_Level1","BN_Module1_Level2");
lgraph = connectLayers(lgraph,"relu_Module1_Level1","concat_1/in1");
lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat_1/in2");
lgraph = connectLayers(lgraph,"concat_1","maxpool_Module1");
lgraph = connectLayers(lgraph,"concat_1","concat1/in1");
lgraph = connectLayers(lgraph,"relu_Module2_Level1","BN_Module2_Level2");
lgraph = connectLayers(lgraph,"relu_Module2_Level1","concat_2/in1");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat_2/in2");
lgraph = connectLayers(lgraph,"concat_2","maxpool_Module2");
lgraph = connectLayers(lgraph,"concat_2","concat2/in1");
lgraph = connectLayers(lgraph,"relu_Module3_Level1","BN_Module3_Level2");
lgraph = connectLayers(lgraph,"relu_Module3_Level1","concat_3/in1");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat_3/in2");
lgraph = connectLayers(lgraph,"concat_3","maxpool_Module3");
lgraph = connectLayers(lgraph,"concat_3","concat3/in1");
lgraph = connectLayers(lgraph,"relu_Module4_Level1","BN_Module4_Level2");
lgraph = connectLayers(lgraph,"relu_Module4_Level1","concat_4/in1");
lgraph = connectLayers(lgraph,"relu_Module4_Level2","concat_4/in2");
lgraph = connectLayers(lgraph,"upsample_Module4","concat3/in2");
lgraph = connectLayers(lgraph,"relu_Module5_Level1","BN_Module5_Level2");
lgraph = connectLayers(lgraph,"relu_Module5_Level1","concat_5/in1");
lgraph = connectLayers(lgraph,"relu_Module5_Level2","concat_5/in2");
lgraph = connectLayers(lgraph,"upsample_Module5","concat2/in2");
lgraph = connectLayers(lgraph,"relu_Module6_Level1","BN_Module6_Level2");
lgraph = connectLayers(lgraph,"relu_Module6_Level1","concat_6/in1");
lgraph = connectLayers(lgraph,"relu_Module6_Level2","concat_6/in2");
lgraph = connectLayers(lgraph,"upsample_Module6","concat1/in2");
lgraph = connectLayers(lgraph,"relu_Module7_Level1","BN_Module7_Level2");
lgraph = connectLayers(lgraph,"relu_Module7_Level1","concat/in2");
lgraph = connectLayers(lgraph,"relu_Module7_Level2","concat/in1");

%% Plot Layers

plot(lgraph);


%% do the training %%
options = trainingOptions('adam', ...
    'MaxEpochs', epnum, ...
    'InitialLearnRate',5e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.95, ...
    'ValidationData',valPatchDs, ...
    'ValidationFrequency',400, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'MiniBatchSize',miniBatchSize);
    
    modelDateTime = datestr(now,'dd-mm-yyyy-HH-MM-SS');
    writematrix([], 'gradient.txt');
    [net,info] = trainNetwork(trPatchDs,lgraph,options);
    save(sprintf(desfold + 'fold_%d-trainedDensenet3d-Epoch-%d.mat', idxFold, epnum),'net');
    
    %Create Graph of learning rate
    iters = length(info.BaseLearnRate);
    epochsp = linspace(1, iters, iters);
    figr = figure;
    rate_plot = plot(epochsp, info.BaseLearnRate);
    title('Learn Rate over Iteration');
    xlabel('Iteration');
    ylabel('Base Learn Rate');
    saveas(figr, dest + 'Rateplot_' + idxFold + '.png', 'png');
    close(2);
    
    %Create graph of gradient
    gradval = load('gradient.txt');
    movefile('gradient.txt', dest + 'gradient_' + idxFold + '.txt');
    iters = length(gradval);
    epochsp = linspace(1, iters, iters);
    figh = figure;
    grad_plot = plot(epochsp, gradval);
    title('Gradient over Iteration');
    xlabel('Iteration');
    ylabel('Gradient');
    saveas(figh, dest + 'Gradplot_' + idxFold + '.png', 'png');
    close(2);    
    
    %Finish saving
    infotable = struct2table(info);
    writetable(infotable, ['fold_' num2str(idxFold) '-Densenet3dinfo-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.txt']);

    close(1)
    h = findall(groot, 'Type', 'Figure'); 
    saveas(h, sprintf(dest + 'Figure_%d.fig', idxFold));
    close(h);
end

