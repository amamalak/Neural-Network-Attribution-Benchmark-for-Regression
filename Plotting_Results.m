
% This script generates the plots of the Figures 3-5 in Mamalakis et al. 2022

% citation: 
% Mamalakis, A., I. Ebert-Uphoff, E.A. Barnes (2022) “Neural network attribution methods for problems
% in geoscience: A novel synthetic benchmark dataset,” arXiv preprint arXiv:2103.10005. 

% Editor: Dr Antonios Mamalakis (amamalak@colostate.edu)

clear 
clc
close all

%% LOAD DATA/RESULTS

% input*gradient for linear model
ncid0= netcdf.open('ItGlin.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'ItGlin');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
ItGlin = d0;
ItGlin=permute(ItGlin,[2,1,3]);
clear d0 ncid0 varid0

% deep taylor
ncid0= netcdf.open('DT.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'DT');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
DT = d0;
DT=permute(DT,[2,1,3]);
clear d0 ncid0 varid0

% LRPab
ncid0= netcdf.open('LRP.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'LRP');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
LRPab = d0;
LRPab=permute(LRPab,[2,1,3]);
clear d0 ncid0 varid0

% LRPz
ncid0= netcdf.open('LRPz.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'LRPz');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
LRPz = d0;
LRPz=permute(LRPz,[2,1,3]);
clear d0 ncid0 varid0

% input*gradient
ncid0= netcdf.open('ItG.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'ItG');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
ItG = d0;
ItG=permute(ItG,[2,1,3]);
clear d0 ncid0 varid0

% integrated gradients
ncid0= netcdf.open('intG.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'intG');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
intG = d0;
intG=permute(intG,[2,1,3]);
clear d0 ncid0 varid0

%gradient
ncid0= netcdf.open('Grad.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'Grad');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
Grad = d0;
Grad=permute(Grad,[2,1,3]);
clear d0 ncid0 varid0

% smooth gradient 
ncid0= netcdf.open('SmooG.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'SmooG');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
SmooG = d0;
SmooG=permute(SmooG,[2,1,3]);
clear d0 ncid0 varid0

% predictions
ncid0= netcdf.open('predictions.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'y_hat_lin');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
y_lin = d0;
clear d0 ncid0 varid0
ncid0= netcdf.open('predictions.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'y_hat_NN');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
y_NN = d0;
clear d0 ncid0 varid0

% occlusion
ncid0= netcdf.open('Occlu.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'Occlu');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
Occlu = d0;
Occlu=permute(Occlu,[2,1,3]);
clear d0 ncid0 varid0

% synthetic data
load('synth_exm_data.mat')
[LON,LAT]=meshgrid(lon,lat);

%% plotting results

load coastlines
coastlon(coastlon<0)=coastlon(coastlon<0)+360;
lat_sst=[89.5:-1:-89.5]';
lon_sst=[0.5:1:359.5]';
[LON_sst,LAT_sst]=meshgrid(lon_sst,lat_sst);

% these figures are used in the paper in Figure 1 (Step 1)
figureHR(lon,lat,SSTrand(:,:,1),lon_sst,lat_sst,1)
colormap(jet)
figureHR(lon,lat,SSTrand(:,:,2),lon_sst,lat_sst,2)
colormap(jet)
figureHR(lon,lat,SSTrand(:,:,end),lon_sst,lat_sst,3)
colormap(jet)

%% XAI RESULTS

close all

% pick a sample of the testing data
t=79476 % corresponds to results in Figure 3 of the paper
t=95903 % corresponds to results in Figure 4 of the paper

% print y(t) and network prediction
y(900000+t)
y_NN(t)

% plot ground truth of attribution
temp=Cnt(:,:,900000+t);
figureHR(lon,lat,temp,lon_sst,lat_sst,4)
title(['Ground Truth of Attribution'])
clear temp


%% PLOT XAI RESULTS FOR THE SELECTED SAMPLE

% Gradient
temp=Grad(:,:,t);
figureHR(lon,lat,temp,lon_sst,lat_sst,5)
%pearson correlation with ground truth
r1=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'rows','complete'); 
%spearman correlation with gound truth
r2=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'Type','Spearman','rows','complete');
title(['NN: Gradient;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
clear temp r1 r2

% Smooth Gradient
temp=SmooG(:,:,t);
figureHR(lon,lat,temp,lon_sst,lat_sst,6)
r1=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'Type','Spearman','rows','complete');
title(['NN: Smooth Gradients;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
clear temp r1 r2

% Input*Gradient
temp=ItG(:,:,t);
figureHR(lon,lat,temp,lon_sst,lat_sst,7)
r1=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'Type','Spearman','rows','complete');
title(['NN: Input*Grad;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
clear temp r1 r2

% Integrated Gradients
temp=intG(:,:,t);
figureHR(lon,lat,temp,lon_sst,lat_sst,8)
r1=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'Type','Spearman','rows','complete');
title(['NN: integrated Gradients;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
clear temp r1 r2

% LRPab
temp=LRPab(:,:,t);
figureHR(lon,lat,temp,lon_sst,lat_sst,9)
% correlation with absolute ground truth
r1=corr(reshape(temp,[],1),abs(reshape(Cnt(:,:,900000+t),[],1)),'rows','complete');
r2=corr(reshape(temp,[],1),abs(reshape(Cnt(:,:,900000+t),[],1)),'Type','Spearman','rows','complete');
title(['NN: LRP_a_1_b_0;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
clear temp r1 r2

% LRPz
temp=LRPz(:,:,t);
figureHR(lon,lat,temp,lon_sst,lat_sst,10)
r1=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'Type','Spearman','rows','complete');
title(['NN: LRP_z;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
clear temp r1 r2

% Occlusion
temp=Occlu(:,:,t);
figureHR(lon,lat,temp,lon_sst,lat_sst,11)
r1=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'rows','complete'); 
r2=corr(reshape(temp,[],1),reshape(Cnt(:,:,900000+t),[],1),'Type','Spearman','rows','complete');
title(['NN: Occlusion;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
clear temp r1 r2

% Deep Taylor
temp=DT(:,:,t);
figureHR(lon,lat,temp,lon_sst,lat_sst,12)
% correlation with absolute ground truth
r1=corr(reshape(temp,[],1),abs(reshape(Cnt(:,:,900000+t),[],1)),'rows','complete');
r2=corr(reshape(temp,[],1),abs(reshape(Cnt(:,:,900000+t),[],1)),'Type','Spearman','rows','complete');
title(['NN: Deep Taylor;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
clear temp r1 r2


%% CORRELATION WITH GROUND TRUTH FOR ALL TESTING SAMPLES

close all

for i=1:length(y_NN) % all testing samples
    
    temp=reshape(Cnt(:,:,900000+i),[],1);
    temp_pos=temp;
    temp_pos(temp_pos<0)=0;
    
    c_ItGlin(i)=corr(temp,reshape(ItGlin(:,:,i),[],1),'rows','complete'); 
    
    c_ItG(i)=corr(temp,reshape(ItG(:,:,i),[],1),'rows','complete');
    c_intG(i)=corr(temp,reshape(intG(:,:,i),[],1),'rows','complete');
    c_Grad(i)=corr(temp,reshape(Grad(:,:,i),[],1),'rows','complete');
    c_SmooG(i)=corr(temp,reshape(SmooG(:,:,i),[],1),'rows','complete');
    c_Occlu(i)= corr(temp,reshape(Occlu(:,:,i),[],1),'rows','complete');
    
    c_LRPz(i)=corr(temp,reshape(LRPz(:,:,i),[],1),'rows','complete');
    
    if y_NN(i)>0
        
    c_LRPab(i)=corr(temp_pos,reshape(LRPab(:,:,i),[],1),'rows','complete');
    c_LRPab2(i)=corr(abs(temp),reshape(LRPab(:,:,i),[],1),'rows','complete');
    
    elseif y_NN(i)<0
      
    c_LRPab(i)=-corr(temp_pos,reshape(LRPab(:,:,i),[],1),'rows','complete');
    c_LRPab2(i)=-corr(abs(temp),reshape(LRPab(:,:,i),[],1),'rows','complete');
     
    
    end
      
    clear temp temp_pos
end


%% PLOTTING LOOP RESULTS
% This is Figure 5 in the paper

figure()
histogram(c_ItGlin, 50, 'Normalization', 'pdf');
hold on
histogram(c_ItG, 50, 'Normalization', 'pdf');
xlim([-.5 1])

figure()
histogram(c_Occlu, 50, 'Normalization', 'pdf');
hold on
histogram(c_ItG, 50, 'Normalization', 'pdf');
hold on
histogram(c_intG, 50, 'Normalization', 'pdf');
hold on
histogram(c_Grad, 50, 'Normalization', 'pdf');
hold on
histogram(c_SmooG, 50, 'Normalization', 'pdf');
xlim([-.5 1])

figure()
histogram(c_LRPab, 50, 'Normalization', 'pdf');
hold on
histogram(c_LRPab2, 50, 'Normalization', 'pdf');
hold on
histogram(c_LRPz, 50, 'Normalization', 'pdf');
xlim([-.5 1])

%%

close all
clc

%%%%%%%%%%%%%%%%% 
%               %
% EXTRA RESULTS %
%               %
%%%%%%%%%%%%%%%%%


%% Investigation of cases with the least aggrement between LRPz and ground truth

close all

id_weak_XAI=find(c_LRPz<0.7); % samples for which LRPz is the least consistent

% regarding y
figure(1)
histogram(y(900001:end), 50, 'Normalization', 'pdf');

figure(2)
histogram(y(900000+id_weak_XAI), 50, 'Normalization', 'pdf');
      

% regarding X
temp=nanmean(SSTrand(:,:,900001:end),3);
figureHR(lon,lat,temp,lon_sst,lat_sst,3)
colormap(jet)
title(['composite SST for all testing samples'])
clear temp

temp=nanmean(SSTrand(:,:,900000+id_weak_XAI),3);
figureHR(lon,lat,temp,lon_sst,lat_sst,4)
colormap(jet)
title(['composite SST for samples that LRPz is the least consistent'])
clear temp

% z statistic
temp=(nanmean(SSTrand(:,:,900000+id_weak_XAI),3)-nanmean(SSTrand(:,:,900001:end),3))./(nanstd(SSTrand(:,:,900001:end),0,3)/sqrt(length(id_weak_XAI)));
figureHR(lon,lat,temp,lon_sst,lat_sst,5)
title(['z statistic for the SST samples that LRPz is the least consistent'])
clear temp


% NOTE: results show that regarding y, nothing special is happening with these samples
% regarding X, these samples seem correpond to extreme X cases. Further
% analysis is needed to estlablish this however. 

%% Check if the benchmark represents climate reality

close all

y_real=y(900001:end)';

figure(1)
histogram(y_real, 50, 'Normalization', 'pdf');
hold on
histogram(y_NN, 50, 'Normalization', 'pdf');
hold off
% Consider composites of the highest and lowest 10% of y to see if they
% represent known climate relationships

temp=nanmean(SSTrand(:,:,900001:end),3);
figureHR(lon,lat,temp,lon_sst,lat_sst,2)
colormap(jet)
title(['composite SST for all testing samples'])
clear temp

figureHR(lon,lat,nanmean(SSTrand(:,:,900000+find(y_real>quantile(y_real,9/10))),3),lon_sst,lat_sst,3)
colormap(jet)
title(['SST composite upper 1/10'])
figureHR(lon,lat,nanmean(SSTrand(:,:,900000+find(y_real<=quantile(y_real,1/10))),3),lon_sst,lat_sst,4)
colormap(jet)
title(['SST composite lower 1/10'])


% z statistics
temp=(nanmean(SSTrand(:,:,900000+find(y_real>quantile(y_real,9/10))),3)-nanmean(SSTrand(:,:,900001:end),3))./(nanstd(SSTrand(:,:,900001:end),0,3)/sqrt(length(find(y_real>quantile(y_real,9/10)))));
figureHR(lon,lat,temp,lon_sst,lat_sst,5)
title(['z statistic for SST composite in upper 1/10'])
clear temp

temp=(nanmean(SSTrand(:,:,900000+find(y_real<=quantile(y_real,1/10))),3)-nanmean(SSTrand(:,:,900001:end),3))./(nanstd(SSTrand(:,:,900001:end),0,3)/sqrt(length(find(y_real<=quantile(y_real,1/10)))));
figureHR(lon,lat,temp,lon_sst,lat_sst,6)
title(['z statistic for SST composite in lower 1/10'])
clear temp

% NOTE: results show a clear ENSO pattern


