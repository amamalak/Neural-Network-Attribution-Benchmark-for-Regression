function []=figureHR(lon,lat,Z,lont,latt,id)
% produce figure in a higher target resolution 
% no interpolation is taking place 

for i=1:length(latt)
    for j=1:length(lont)
        [~,latid]=min(abs((latt(i)-lat)));
        [~,lonid]=min(abs((lont(j)-lon)));
        ZZ(i,j)=Z(latid,lonid);
        
        clear latid lonid
    end
end


lim=max(max(abs(ZZ)));


%
load coastlines
coastlon(coastlon<0)=coastlon(coastlon<0)+360;

%[LONt,LATt]=meshgrid(lont,latt);

figure(id)
axesm eckert4
contourfm(latt,lont,ZZ,-lim:2*lim/20:lim,'LineStyle','none')
hold on
geoshow('landareas.shp','FaceColor','black') 
caxis([-lim lim])
colormap(bluewhitered)
%colormap(jet)
contourcbar