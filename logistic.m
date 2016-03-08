function main
clear;
%dataset = 'mushroom';
%dataset = 'liver';
%dataset = 'isolet';
dataset = 'ionosphere';



load([dataset, '_train']);
size_X = size(X);
w = rand(1,size_X(2)+1)*0.02-0.01;

%% fminunc function
w_train = fminunc(@(w)costFunction(w,X,Y),w)

%% gradient-descent Algorithm
%w_train = trainFunction(w,X,Y)

%%  train ratio
g = w_train * [X ones(size_X(1),1)]';
y_train = 1./(1+exp(-g));
result_train = round(y_train') - Y;
result_ration = 1 - sum(abs(result_train)) / size_X(1) 

%% test ratio
load([dataset, '_test']);
size_X = size(X);
g = w_train * [X ones(size_X(1),1)]';
y_test = 1./(1+exp(-g));
result_test = round(y_test') - Y;
result_ration = 1 - sum(abs(result_test)) / size_X(1) 


end

function cost = costFunction(w,X,Y)
size_X = size(X);
a = w * [X ones(size_X(1),1)]';
y = 1 ./ [ 1.+ exp(-a)];
cost = -(log(y) * Y + log(1-y)*(1.- Y));
end

function w_train = trainFunction(w,X,Y)
size_X = size(X);
step = 0.001;
cnt = 2000;
i = 0;
while(i<cnt)
    o = w * [X ones(size_X(1),1)]';
    y = 1./(1+exp(-o));
    dw = (Y' -y) * [X ones(size_X(1),1)];
    w = w + step * dw;
    i = i+1;
end

w_train = w;

end


