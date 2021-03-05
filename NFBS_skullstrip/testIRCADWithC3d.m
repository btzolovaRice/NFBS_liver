% Segmentation on Test Data

% Clear workspace
clear; close all; clc;

destination_runs = pwd + "/IRCADwithC3d";
destination = pwd + "/testrun";

imgDir = dir(fullfile(destination, 'patient_CT','*.nii'));
imgFile = {imgDir.name}';
imgFolder = {imgDir.folder}';

lblDir = dir(fullfile(destination, 'vessel_lbl','*.nii'));
lblFile = {lblDir.name}';
lblFolder = {lblDir.folder}';

%%Load test indices 
s = load('idxTest.mat');
c = struct2cell(s);
idxTest = cat(1,c{:});

%Load Patient id
vName = 'inputIRCAD.json';
jsonData = jsondecode(fileread(vName));
fullFileName = jsonData.fullFileName;
delimiter = jsonData.delimiter;
T = readtable(fullFileName, 'Delimiter', delimiter);
A = table2array(T);
idCol = jsonData.idCol;
PId = A(:,idCol);
patientId = cellstr(PId);

C = cell(25,5);
[testPatientId, imgFileTest, imgFolderTest, lblFileTest, lblFolderTest] = deal(C);

for kfold = 1:1
    
    disp(['Processing K-fold-' num2str(kfold)]);
    
    trainedNetName = ['fold_' num2str(kfold) '-trainedDensenet3d-Epoch-5.mat'];
    load(fullfile(destination_runs, trainedNetName));
          
    testSet = idxTest{1,kfold};
    testPatientId(:,kfold) =  PId(testSet);%create test patientid set
    save('testPatientId.mat','testPatientId');
    
    imgFileTest(:,kfold) = imgFile(testSet);
    imgFolderTest(:,kfold) = imgFolder(testSet);
    
    lblFileTest(:,kfold) = lblFile(testSet);
    lblFolderTest(:,kfold) = lblFolder(testSet);
   
    %create directories to store labels 
        mkdir(fullfile(destination_runs,['predictedLabel-fold' num2str(kfold)]));
        mkdir(fullfile(destination_runs,['groundTruthLabel-fold' num2str(kfold)]));
        
    for id = 1:length(imgFileTest)
        
        imgLoc = fullfile(imgFolderTest(id,kfold),imgFileTest(id,kfold));
        imgName = niftiread(char(imgLoc));
        imginfo = niftiinfo(char(imgLoc));
               
        lblLoc = fullfile(lblFolderTest(id,kfold),lblFileTest(id,kfold));
        lblName = niftiread(char(lblLoc));
        lblinfo = niftiinfo(char(lblLoc));
       
        patientId = char(testPatientId(id,kfold));
               
        predLblName = ['predictedLbl_', patientId];
        grdLblName = ['groundTruthLbl_',patientId];

        predDir = fullfile(destination_runs,['predictedLabel-fold' num2str(kfold)],predLblName);
        groundDir = fullfile(destination_runs,['groundTruthLabel-fold' num2str(kfold)],grdLblName);

        groundTruthLabel = lblName;
        fprintf('line78');
        predictedLabel = semanticseg(imgName,net,'ExecutionEnvironment','cpu');
        fprintf('line82\n');
        
        % save preprocessed data to folders
        niftiwrite(single(predictedLabel),predDir) %,imginfo);
        niftiwrite(groundTruthLabel,groundDir,lblinfo);
                           
    end
end