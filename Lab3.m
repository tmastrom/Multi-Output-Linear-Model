
% training dataset
Xtr = D_build_tr(1:8,:);    % measurements
Ytr = D_build_tr(9:10,:);   % labels

% testing dataset
Xte = D_build_te(1:8,:);    % measurements
Yte = D_build_te(9:10,:);   % labels

% prepare matrix X_hat from Xtr using Eq. 3.2
Xhat = [Xtr', ones(640,1)]; 

% find weight and bias vectors using Eq. 3.3
eps = 0.01; % small scalar 
W = (Xhat'*Xhat + eps*eye(9))\Xhat'*Ytr'; % Eq. 3.3

% using Eq. 3.4
Xhatp = pinv(Xhat); % psuedo-inverse of Xhat
W_ = Xhatp*Ytr';


% Apply model to the test data
Yp = W(1:8,:)'*Xte + [b1;b2];   % Eq.3.3 results
err = (norm( (Yte - Yp)  , 'fro')/ norm(Yte, 'fro'));

Yp_ = W_(1:8,:)'*Xte + [b1_;b2_]; % Eq. 3.4 results
err_ = (norm( (Yte - Yp_)  , 'fro')/ norm(Yte, 'fro'));

% plot the first row of the test and predicted values
x = linspace(0,max(max(Yte(1,:)',Yp(1,:)')), 128);
figure('name','Heating Values')
plot(x,Yte(1,:),'k',x, Yp(1,:),'r')

% plot 2nd row of test and predicted values
x1 = linspace(0,max(max(Yte(2,:)',Yp(2,:)')), 128);
figure('name','Cooling Values')
plot(x,Yte(2,:),'k',x, Yp(2,:),'b')



