
% This script performs a Monte Carlo simulation to determine the effect of
% the number of break points (K) of a piece wise linear function (that is
% randomly built) on the degree of linearity of the function.
% The result is that that the higher the value of K, the more unlikely it is 
% to obtain approximately a linear function (see Mamalakis et al. 2022)

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

%% Generate synthetic inputs X and outputs y for different values of K

close all

% SST vector
SSTv=reshape(SSTanom,[],size(SSTanom,3),1);
% position of ocean grid points in the vector
ID_sea=find(isnan(SSTv(:,1))==0);

S=corrcoef(SSTv(ID_sea,:)'); % define spatial dependence structure (covariance matrix)

N=5000; % number of samples in each synthetic dataset

i=0;
for k=[0,1,2,3,5,10,15,20,30,50] % different values of K
    i=i+1;
    k % printing K value in the loop
    for j=1:10
        %generate random SST (there is no memory)
        SSTv_rand=nan*ones(size(SSTv,1),N);
        SSTv_rand(ID_sea,:)=(mvnrnd(zeros(length(ID_sea),1),S,N))';
        
        [y,W,C]=PWL_gen(SSTv_rand,k,1); % generate synthetic output y
        
        % determine the linearity in each pixel
        C_linear=nan*C;

        for ii=1:length(ID_sea) % for each pixel 

           b = regress(C(ID_sea(ii),:)',[ones(size(C,2),1),SSTv_rand(ID_sea(ii),:)']);
           C_linear(ID_sea(ii),:)=[ones(size(C,2),1),SSTv_rand(ID_sea(ii),:)']*b; 
           
           clear b
        end
        
        % the varaince of the true piece-wise linear function that is
        % explained by a line model (this is a proxy of the degree of
        % linearity)
        R2(:,j,i)=1-mean((C-C_linear).^2,2)./mean((C-repmat(mean(C,2),1,size(C,2))).^2,2);
        
        clear SSTv_rand W C y C_linear
    end
    
end

%% Plotting results of the simulation

figure()
plot(reshape(repmat([0,1,2,3,5,10,15,20,30,50],size(R2,2),1),[],1),reshape(permute(nanmean(R2),[2,3,1]),[],1),'o')

figure()
plot(reshape(repmat([0,1,2,3,5,10,15,20,30,50],size(R2,2),1),[],1),reshape(permute(nanstd(R2),[2,3,1]),[],1),'o')

figure()
hist(reshape(R2(:,1,2),[],1)) % K= 1
title(['mean = ', num2str(nanmean(R2(:,1,2)))])

figure()
hist(reshape(R2(:,1,5),[],1)) % K= 5
title(['mean = ', num2str(nanmean(R2(:,1,5)))])

