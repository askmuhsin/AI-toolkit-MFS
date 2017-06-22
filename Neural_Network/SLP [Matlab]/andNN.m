
clear all
clc
%-------------------------------------------------------------%
% inputs / hyperparameters
x1 = [0,0,1,1];
x2 = [0,1,0,1];
%y  = [0,1,1,1]; %output / training data for OR gate
y = [0,0,0,1]; %training data for AND gate
w = [0,0,0]; %weights initialisation, w(0) is bias

l = 1; %learning rate
u = 0; %perceptron output
ydash = 0; %output after activations ydash = signumfn(u)
flag = 0; %flag will be incremented when error for a given epoch is 0
run = 1; %when flag is counted twice, run will be 0, to end training
epoch = 0;%counter for epoch

%-------------------------------------------------------------%
%backpropagation / training
while (run==1)
    dw = 0;    
    for i = 1:4
        u=(x1(i)*w(2))+(x2(i)*w(3))+w(1);
        ydash = signumfn(u);
        dw1 = 1 * (y(i)-ydash);
        dw2 = l * (y(i)-ydash)*x1(i);
        dw3 = l * (y(i)-ydash)*x2(i);
        
        w(1) = w(1) + dw1;
        w(2) = w(2) + dw2;
        w(3) = w(3) + dw3;
        
        dw = dw + (y(i)-ydash);
    end
    
    %if error continues to stay 0 for three consecutive epochs stop training
    dw = dw /4;
    epoch  = epoch + 1;
    fprintf('Iteration %d, error value = %d\n',epoch,dw)
    if (dw ~= 0)
        flag = 0;
    end
    if (dw == 0)
        flag = flag + 1;
    end
    if (flag==3)
        run = 0; %on thrid consecutive 0 error, stop running (training)
    end
             
end

%-------------------------------------------------------------%
%feed forward network
for j = 1:4
    u=(x1(j)*w(2))+(x2(j)*w(3))+w(1);
    ydash = signumfn(u);
    fprintf('%d %d : %d \n',x1(j),x2(j),ydash);
end
