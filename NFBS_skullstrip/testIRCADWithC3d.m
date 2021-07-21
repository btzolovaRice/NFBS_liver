% Segmentation on Test Data

% Clear workspace
clear; close all; clc;
kfoldnum = zeros(0,1); mydice = zeros(0,1); matdice = zeros(0,1);

% Call the directories
destination_runs = pwd + "/IRCADwithC3d/liver_40epoch_13";
destination = pwd + "/testrun/"; %image files
epnum = 40;

imgDir = dir(fullfile(destination, 'background_null/patient_CT','*.nii'));
imgFile = {imgDir.name}';
imgFolder = {imgDir.folder}';

lblDir = dir(fullfile(destination, 'background_null/vessel_lbl','*.nii'));
lblFile = {lblDir.name}';
lblFolder = {lblDir.folder}';

%%Load test indices 
s = load(destination_runs + '/idxTest.mat');
c = struct2cell(s);
idxTest = cat(1,c{:});

%Load Patient id
vName = 'inputIRCAD_nullback.json';
jsonData = jsondecode(fileread(vName));
fullFileName = jsonData.fullFileName;
delimiter = jsonData.delimiter;
T = readtable(fullFileName, 'Delimiter', delimiter);
A = table2array(T);
idCol = jsonData.idCol;
PId = A(:,idCol);
patientId = cellstr(PId);

C = cell(1,5);
[testPatientId, imgFileTest, imgFolderTest, lblFileTest, lblFolderTest] = deal(C);

for kfold = 1:length(idxTest)
    
    disp(['Processing K-fold-' num2str(kfold)]);
    
    trainedNetName = ['fold_' num2str(kfold) '-trainedDensenet3d-Epoch-' num2str(epnum) '.mat'];
    load(fullfile(destination_runs, trainedNetName));
          
    testSet = idxTest{1,kfold};
    testPatientId(:,kfold) =  PId(testSet); %create test patientid set
    save('testPatientId.mat','testPatientId');
    
    imgFileTest(:,kfold) = imgFile(testSet);
    imgFolderTest(:,kfold) = imgFolder(testSet);
    
    lblFileTest(:,kfold) = lblFile(testSet);
    lblFolderTest(:,kfold) = lblFolder(testSet);
   
    %create directories to store labels 
        mkdir(fullfile(destination_runs,['predictedLabel-fold' num2str(kfold)]));
        mkdir(fullfile(destination_runs,['groundTruthLabel-fold' num2str(kfold)]));
        
    for id = 1:size(imgFileTest,1)
        
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
        predictedLabel = semanticseg(imgName,net,'ExecutionEnvironment','cpu');
        
        % save preprocessed data to folders
        niftiwrite(single(predictedLabel),predDir,imginfo);
        niftiwrite(groundTruthLabel,groundDir,lblinfo);
                       
    end
    
    %shift predicted image so the values are 0 and 1       
    predL = [predDir + '.nii'];    
    system(sprintf('c3d %s -shift -1 -o %s', predL, predL));     

    PL = niftiread(predL);
    GT = niftiread(groundDir + '.nii');

    
    % calculate the dice score and save
    [diceval, dicemat] = dicescorecalc(PL, GT);
    kfoldnum(end+1, 1) = kfold;
    mydice(end+1, 1) = diceval;
    matdice(end+1, 1) = dicemat;
    fprintf('The dice score is %d\n', diceval);
    
end

T = table(kfoldnum, mydice);
T.Properties.VariableNames = {'K fold', 'Dice Score', 'Matlab Dice'};
writetable(T, destination_runs + "/dicevalues.txt");