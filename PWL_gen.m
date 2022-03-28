function [ysim,W,C]=PWL_gen(X,K,N) %[ysim,W,C,WW]=PWL_gen(X,K,N)

% Piece Wise Linear generation of synthetic series
% 
% X: the synthetic input in D (pixels) by T (samples)
% K: number of break points in the piece wise linear function (scalar);
%    the point (0,0) is always a break point
% N: number of generated series (realizations)
%
% ysim: the syntetic series in N (number of realizations) by T (samples in each series)
% W: the weight vector for each linear piece and for each realization in D
%    by K+1 by N
% C: the actual contibution of each pixel to y value in D by T by N
%
% WW (if uncommented): the actual gradient of the contribution of each pixel to y (gradient of C) in D by T by N 

ID_nonan=find(isnan(X(:,1))==0);
S=corrcoef(X(ID_nonan,:)');

W=nan*ones(size(X,1),K+1,N);
C=nan*ones(size(X,1),size(X,2),N);
%WW=nan*ones(size(X,1),size(X,2),N);

for nn=1:N

    for i=1:K+1
    % random weights
    W(ID_nonan,i,nn)=(mvnrnd(zeros(length(ID_nonan),1),S,1))';
    end
    
    % random break points with spatial structure as the one embedded in X
    lq=(mvnrnd(zeros(length(ID_nonan),1),S,K-1))';
    lq=normcdf(lq,0,1);

    for i=1:length(ID_nonan) % for each pixel 
       
       C(ID_nonan(i),:,nn)=0;
       temp=X(ID_nonan(i),:);
       
       if K==0 % trivial case of a purely linear model
       C(ID_nonan(i),:,nn)=W(ID_nonan(i),1,nn).*temp;     
%      WW(ID_nonan(i),:,nn)=W(ID_nonan(i),1,nn);
       
       elseif K>0
           
       [f1,y1]=ecdf(temp');
       f1(1)=[]; y1(1)=[];
       l=interp1(f1,y1,lq(i,:)','linear','extrap'); % evaluate the break points for this pixel
       id_zero=sum(l<0)+1; % the rank of 0 in the list of the breal points

       edges=[-Inf;sort([l;0]);Inf];

       Y=discretize(temp',edges);
%      WW(ID_nonan(i),:,nn)=W(ID_nonan(i),Y',nn);

       for yy=id_zero:-1:1 %negative break-points
       id=find(Y<=yy);
       C(ID_nonan(i),id,nn)=C(ID_nonan(i),id,nn)+W(ID_nonan(i),yy,nn).*max(temp(id)-edges(yy+1),edges(yy)-edges(yy+1));
       clear id
       end

       for yy=id_zero+1:1:max(Y) %positive break-points
       id=find(Y>=yy);
       C(ID_nonan(i),id,nn)=C(ID_nonan(i),id,nn)+W(ID_nonan(i),yy,nn).*min(temp(id)-edges(yy),edges(yy+1)-edges(yy));
       clear id
       end

       clear f1 y1 l id_zero edges Y
       
       end
        
       clear temp
       
    end
    
    clear lq

    ysim(nn,:)=nansum(C(:,:,nn),1)/length(ID_nonan); 
    C(:,:,nn)=C(:,:,nn)/length(ID_nonan);
    W(:,:,nn)=W(:,:,nn)/length(ID_nonan);
%   WW(:,:,nn)=WW(:,:,nn)/length(ID_nonan);
    
end

