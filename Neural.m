function NeuralNetwork

clear;
dataset = 'data/mushroom';
%dataset = 'data/liver';
%dataset = 'data/isolet';
%dataset = 'data/ionosphere';



load([dataset, '_train']);
size_X = size(X);
X_=[X ones(size_X(1),1)];
[N,d] = size(X_);
Eta = 0.1;
iter_cont = 10;
h_cont = 5;

%%Eta change with iter_cont

for h = 2:20
    
    %%training
    ratio = 0;
    for j = 1 : h_cont
        remain = randsample(N,round(N*0.2));
        test_X = X_(remain,:);
        test_Y = Y(remain,:);        
        mask = setdiff(1:N,remain);
        train_X = X_(mask,:);
        train_Y = Y(mask,:);
        [v w] = training(train_X,train_Y,Eta,iter_cont,h);
        %% check training result
        right_num = 0;
        for i = 1 :size(test_X,1)          
            zh = test_X(i,:) * w;
            zh = 1./(1.+exp(-zh));
            zh_ag = [zh 1];
            y = zh_ag * v;
            %y = round(y);
            y = round(1./(1.+exp(-y)));
            if y == test_Y(i)
                right_num = right_num +1;
            end
        end
        ratio = ratio + right_num / size(test_X,1);
        
    end
    h
    ratio = ratio / h_cont
    
end

end

function [v w]=training(X,Y,Eta,iter_cont,h)
w = rand(size(X,2),h)*0.02-0.01;
v = rand(h+1,1)*0.02-0.01;

for i = 1:iter_cont
    error=0;
    for j = 1:size(X,1)
        zh = X(j,:) * w;
        zh = 1./(1.+exp(-zh));
        zh_ag = [zh 1];
        y = zh_ag * v;
        y = 1./(1.+exp(-y));
        dv = Eta * (Y(j)-y)*zh_ag';
        v_dg = v;
        v_dg(end,:)=[];
        dw = Eta * (Y(j)-y) * X(j,:)'*(zh.*zh.*(1.-zh).*v_dg');
        v= v + dv;
        w = w + dw;
    end
% 
end

end
