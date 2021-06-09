classdef dicePixelClassification3dLayer < nnet.layer.ClassificationLayer
    % This layer implements the generalized dice loss function for training
    % semantic segmentation networks.
    
    % References
    % ----------
    % Sudre, Carole H., et al. "Generalised Dice overlap as a deep learning
    % loss function for highly unbalanced segmentations." Deep Learning in
    % Medical Image Analysis and Multimodal Learning for Clinical Decision
    % Support. Springer, Cham, 2017. 240-248.
    %
    % Copyright 2018 The MathWorks, Inc.
    
    properties(Constant)
        Epsilon = 1e-8; %small change to prevent division by 0 
        mbsize = 4; %the minibatchsize
        W1 = 1.6; %weight for the vessel
        W2 = 0.4; %weight for the background
    end
    
    methods
        
        function layer = dicePixelClassification3dLayer(name)
            % layer =  dicePixelClassification3dLayer(name) creates a Dice
            % pixel classification layer with the specified name.
            
            % Set layer name.          
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'Dice loss';
        end
        
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Dice loss between
            % the predictions Y and the training targets T.   

            % Weights by inverse of region size.
            summation = sum(sum(sum(T,1),2),3).^2;
            if max(eps, summation) == eps 
                error('Error. Summation must be greater than machine error.');
            else 
                W = 1./max(eps, summation);
            end
            
            % over spatial dimensions 1,2,3
            intersection = sum(sum(sum(Y.*T,1),2),3);
            union = sum(sum(sum(Y.^2 + T.^2, 1),2),3);          
            
            %Numerator = zeros(layer.mbsize, 1);
            %Denominator = zeros(layer.mbsize, 1);
            
            for i=1:4
                Numerator = layer.W1*W(1,1,1,1,i).*intersection(1,1,1,1,i) + layer.W2*W(1,1,1,2,i).*intersection(1,1,1,2,i);
                Denominator = layer.W1*W(1,1,1,1,i).*union(1,1,1,1,i) + layer.W2*W(1,1,1,2,i).*union(1,1,1,2,i);
            end 
            
            % over channels dim (4) :-  representing classes
            numer = 2*Numerator + layer.Epsilon;
            denom = Denominator + layer.Epsilon;
            
            % Compute Dice score.
            dice = numer./denom;
            
            % Return average Dice loss over minibatch (5th dim).
            N = size(Y,5);
            loss = sum((1-dice))/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer, Y, T) returns the derivatives of
            % the Dice loss with respect to the predictions Y.
            
            % Weights by inverse of region size.
            W = 1./ max(eps,sum(sum(sum(T,1),2),3).^2);
            
            intersection = sum(sum(sum(Y.*T,1),2),3);
            union = sum(sum(sum(Y.^2 + T.^2, 1),2),3);
     
            numer = 2*sum(W.*intersection,4) + layer.Epsilon;
            denom = sum(W.*union,4) + layer.Epsilon;
            
            N = size(Y,5);
            dLdY = (2*W.*Y.*numer./denom.^2 - 2*W.*T./denom)./N;
            %fprintf('%1.2e\n', dLdY); 
            %grad_val(end+1) = dLdY; 
        end
    end
end

