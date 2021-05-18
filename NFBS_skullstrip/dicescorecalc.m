%Upload images 
predL = niftiread('ptesting.nii');
groundT = niftiread('gtesting.nii');

%Change image values to be 0 and 1 
system(sprintf('c3d %s -shift -1 -o %s', predL, predL));  
system(sprintf('c3d %s -replace 60 1 -o %s', groundT, groundT));  

%double
predLabel = double(predL);
groundTruth = double(groundT);

sdice = dice(predLabel, groundTruth); 

[loss, diceval] = forwardLoss(predLabel, groundTruth);

function [loss, diceval] = forwardLoss(Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Dice loss between
            % the predictions Y and the training targets T.   
            Epsilon = 1e-8;
            
            % Weights by inverse of region size.
            W = 1./(sum(sum(sum(T,1),2),3).^2); 
                %this lines up with our expected value 
            
            % over spatial dimensions 1,2,3
            intersection = sum(sum(sum(Y.*T,1),2),3);
            union = sum(sum(sum(Y.^2 + T.^2, 1),2),3);          
            
            % over channels dim (4) :-  representing classes
            numer = 2*W.*intersection + Epsilon;
            denom = W.*union + Epsilon;
            
            % Compute Dice score.
            diceval= numer./denom;
            
            % Return average Dice loss over minibatch (5th dim).
            loss = (1-diceval);
        end