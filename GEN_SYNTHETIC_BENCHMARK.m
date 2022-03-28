
% This script generates a piece-wise linear synthetic attribution benchmark
% as introduced in Mamalakis et al. 2022
% NOTE: if this script is run it will produce a different benchmark than
% the one in paper (due a different seed in the random generator)

% citation: 
% Mamalakis, A., I. Ebert-Uphoff, E.A. Barnes (2022) “Neural network attribution methods for problems
% in geoscience: A novel synthetic benchmark dataset,” arXiv preprint arXiv:2103.10005. 

% Editor: Dr Antonios Mamalakis (amamalak@colostate.edu)

clear 
clc
close all

%% SST data from the COBESST2 dataset at https://www.esrl.noaa.gov/psd/data/gridded/data.cobe2.html

% units: Celsius, period: 01/1850 - 12/2019
ncid0= netcdf.open('sst.mon.mean_cobe_v2.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'sst');
d0=netcdf.getVar(ncid0,varid0,'double');

varid_lat= netcdf.inqVarID(ncid0,'lat');
lat_sst=netcdf.getVar(ncid0,varid_lat,'double');
varid_lon= netcdf.inqVarID(ncid0,'lon');
lon_sst=netcdf.getVar(ncid0,varid_lon,'double');
[LON_sst,LAT_sst]=meshgrid(lon_sst,lat_sst);
netcdf.close(ncid0);

monthlySST = d0;
clear d0
monthlySST=permute(monthlySST,[2,1,3]); % restructuring data array
for i=1:length(lat_sst)
    for j=1:length(lon_sst)
        
monthlySST(i,j,abs(monthlySST(i,j,:))>500)=NaN; % determining the NaN values

    end
end

% load matlab coastlines
load coastlines
coastlon(coastlon<0)=coastlon(coastlon<0)+360;

% double check you read the data consistently by comparing it with plots from the website!!
% This figure is not used in the paper
figure()
contourf(LON_sst,LAT_sst,monthlySST(:,:,end-12))% 12/2018
hold on 
plot(coastlon,coastlat,'.')

max(max(monthlySST(:,:,end-12)))% 12/2018
min(min(monthlySST(:,:,end-12)))% 12/2018

% Now keep only the data in 01/1950-12/2019 (we do not trust earlier data)
temp=monthlySST;
clear monthlySST
monthlySST=temp(:,:,100*12+1:end);
clear temp

%% anomalies & detrending

%get monthly anomalies (remove annual cycle)
for i=1:12
 monthlySST(:,:,i:12:end)= monthlySST(:,:,i:12:end)-repmat(nanmean(monthlySST(:,:,i:12:end),3),[1,1,size(monthlySST(:,:,i:12:end),3)]);       
end

%detrending
for i=1:length(lat_sst)
    for j=1:length(lon_sst)
        temp=squeeze(monthlySST(i,j,:));
        temp=detrend(temp); %detrending
        SSTanom(i,j,:)=temp/std(temp); %standardized
        clear temp 
    end
end

% plot again the same map but after detrending and standardizing
% This figure is not used in the paper
figure()
contourf(LON_sst,LAT_sst,SSTanom(:,:,end-12))% 12/2018
hold on 
plot(coastlon,coastlat,'.')

%% Reduce dimensions: Use a grid of 10 by 10 degrees.

rl=10;

temp = SSTanom;
clear SSTanom
SSTanom=temp(1:rl:end,1:rl:end,:);
clear temp
lat=lat_sst(1:rl:end);
lon=lon_sst(1:rl:end);
[LON,LAT]=meshgrid(lon,lat);

%% Generate Synthetic Random Input X

close all

% SST vector
SSTv=reshape(SSTanom,[],size(SSTanom,3),1);
% position of ocean grid points in the vector
ID_sea=find(isnan(SSTv(:,1))==0);

%generate random SST (there is no memory)
N=1000000; %number of random fields
SSTv_rand=nan*ones(size(SSTv,1),N);

S=corrcoef(SSTv(ID_sea,:)'); % define spatial dependence structure (covariance matrix)

SSTv_rand(ID_sea,:)=(mvnrnd(zeros(length(ID_sea),1),S,N))'; % generate random SST
% NOTE: the generated fields are not the same with the ones of the paper
% since a different seed of random randoms is used each time 
% this code can nevertheless be used to genreate synthetic benchmarks 

%% Plot some maps of the synthetic SSTs just for visual inspection

% these figures are used in the paper in Figure 1 (Step 1)
% AGAIN: the produced figures are not the same with the ones in the paper
% since the generated synthetic data are different (i.e. they come from a
% different random seed)

% first sample
figureHR(lon,lat,reshape(SSTv_rand(:,1),size(SSTanom,1),size(SSTanom,2)),lon_sst,lat_sst,1)
colormap(jet)
% second sample
figureHR(lon,lat,reshape(SSTv_rand(:,2),size(SSTanom,1),size(SSTanom,2)),lon_sst,lat_sst,2)
colormap(jet)
% last sample
figureHR(lon,lat,reshape(SSTv_rand(:,end),size(SSTanom,1),size(SSTanom,2)),lon_sst,lat_sst,3)
colormap(jet)

%% Generate Synthetic Response Y 

% this function generates a random piece wise linear response Y to X
K=5;
[y,W,C]=PWL_gen(SSTv_rand,K,1);  

%%%%%% Uncomment to use %%%%%
% make median equal to zero (modify if you want the mean to be zero)
%C=C-repmat(permute(nanmedian(y,2),[3,2,1]),size(C,1),size(C,2),1)/length(ID_sea); 
%y=y-repmat(nanmedian(y,2),1,size(y,2));
% NOTE: If you use the above commands, then the piece wise linear function in each pixel is NOT crossing the
% (0,0) point anymore; It is crossing the point (0,median(y)/#pixels)
% for the paper we did NOT use the above commands

% map of weights of the piece-wise linear function 
% this figure produces a version of figure 2 (top) in the paper
figureHR(lon,lat,reshape(W(:,end),size(SSTanom,1),size(SSTanom,2)),lon_sst,lat_sst,4)
colormap(jet)

%adding noise to the output (uncomment if you want to add noise)
%y_obs1=y+norminv(rand(1,length(y)),0,std(y)/2);

%print the synthetic series and some statistics for inspection
figure()
plot(y,'b') % similar to Figure 1 Step 2 of the paper

figure()
hist(y) % not used in the paper

figure()
autocorr(y) % not used in the paper

sum(y>0)
sum(y<0)
sum(y==0)

mean(y)
var(y)

%%  We assume a linear model to predict y given X

x=SSTv_rand(ID_sea,1:900000)';
b= (x'*x)\x'*y(1:900000)'; %least squares
clear x
x=SSTv_rand(ID_sea,900001:end)';
% Variance explained
R2_ls=1-sum((y(900001:end)'-x*b).^2)/sum((y(900001:end)'-mean(y(900001:end)')).^2)
clear b x

%% Examples of piece-wise linear functions

% pick a point in the globe (user defined)
lat1=-50;
lon1=50;

[~,Ilat]=min(abs(lat-lat1));
[~,Ilon]=min(abs(lon-lon1));
id=find((reshape(LAT,[],1)==lat(Ilat)).*(reshape(LON,[],1)==lon(Ilon))==1);

% similar to Figure 2 (bottom) in the paper
figure ()
plot(SSTv_rand(id,:)',C(id,:)', 'ro')
clear id

figure()
plot(lon(Ilon),lat(Ilat),'r*')
hold on 
plot(coastlon,coastlat,'.')


%% saving the synthetic benchmark
% reformatting from a vector to a map

% synthetic input
SSTrand=reshape(SSTv_rand,size(SSTanom,1),size(SSTanom,2),[]);
% ground truth of attribution
Cnt=reshape(C,size(SSTanom,1),size(SSTanom,2),[]);
% weights
temp=reshape(W,size(SSTanom,1),size(SSTanom,2),[]);
clear W
W=temp;
clear temp

save('synth_exm_data.mat','y','SSTrand','lon','lat','W','Cnt','-v7.3')

