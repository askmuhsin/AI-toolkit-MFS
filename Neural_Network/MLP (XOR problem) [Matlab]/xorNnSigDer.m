clear all
clc
%-----------------------------------------------
input  = [0 0 ; 0 1; 1 0 ; 1 1];
output = [0 ; 1 ; 1 ; 0] ;

wi = rand(2,4); %weights for layer 1 (bw input and hidden)
wh = rand(4,1); %weights for layer 2 (bw hidden and output)

s1 = sigmoid(input, wi);%s1 = 1./(1+exp(-(input* wi)))
s2 = sigmoid(s1, wh);
fprintf('Before Training : \n');
disp(s2)
epoch = 60000;


for i = 1:epoch
    l0 = input;
    l1 = sigmoid(input, wi);
    l2 = sigmoid(s1, wh);
    
    el2 = output - l2; %error of l2 (bw hidden and out)
    dl2 = el2 .* sigmoid_der(l2);
    
    el1 = dl2 * wh';
    dl1 = el1 .* sigmoid_der(l1);
    
    wh = wh + (l1' * dl2); %update weight
    wi = wi + (l0' * dl1); 
    epoch = epoch + 1;
end

%%------------------------------------------------------
fprintf('\nOptimized weights wi : \n')
disp(wi)
fprintf('\nOptimized weights wh : \n')
disp(wh)

fprintf('\nInput : \n')
disp(input)
fprintf('\nExpected Output : \n')
disp(output)
fprintf('\nActual Output : \n')

s1 = sigmoid(input, wi);%s1 = 1./(1+exp(-(input* wi)))
s2 = sigmoid(s1, wh);
disp(s2)