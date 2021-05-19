classdef dicescorecalc
    methods
        function [diceval, dicemat] = DiceValueCal(Y, T)
                    % loss = forwardLoss(layer, Y, T) returns the Dice loss between
                    % the predictions Y and the training targets T. 
                    Y = double(Y);
                    T = double(T);
                    
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
                    diceval= numer./denom; %dice score manually calculated
                    
                    dicemat=dice(Y,T); %dice score calc by matlab 
        end
    end
end 