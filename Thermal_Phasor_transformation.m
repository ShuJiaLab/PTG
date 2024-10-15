
% *************************************************************************
%    Phasor Thermography: Thermal Phasor Transformation
%                           
%
% Author: Dingding Han
% Contact: dhan314@gatech.edu
%
% Copyright (c) 2024 Dingding Han
% All rights reserved.
%
% This software is provided "as-is," without any express or implied
% warranty. In no event shall the author be held liable for any damages
% arising from the use of this software.
%
% Permission is granted to anyone to use this software for any purpose,
% including commercial applications, and to alter it and redistribute it
% freely, subject to the following restrictions:
%
% 1. The origin of this software must not be misrepresented; you must not
%    claim that you wrote the original software. If you use this software
%    in a product, an acknowledgment in the product documentation would
%    be appreciated but is not required.
%
% 2. Altered source versions must be plainly marked as such, and must not
%    be misrepresented as being the original software.
%
% 3. This notice may not be removed or altered from any source
%    distribution.
% *************************************************************************
%% phasor transformation using fft 



%% data load
dimensions =[451,481,10];
myFilename='motorcorey_regi.bsq';
precisionOfData='double';
readOffset =0;
interleaveType = 'bsq';
byteOrdering = 'ieee-le';

Data_regi = multibandread('motorcorey_regi.bsq',dimensions,precisionOfData,readOffset,interleaveType,byteOrdering);
save  'Phasor_data/Data_regi.mat' Data_regi
duckim=adapthisteq(rescale(Data_regi(:,:,1),0,1));
figure;
imshow(duckim);




% Datara = multibandread('facera.bsq',dimensions,precisionOfData,readOffset,interleaveType,byteOrdering);
% save  'Phasor_data/Datacount.mat' Datara

%% Fourier transform 

img_width=size(Data_raw, 2);
img_height=size(Data_raw,1);
band_num=size(Data_raw,3);
fre_num=band_num;
Ca=zeros(band_num,1);



Radiance=zeros(band_num,1);
%p0=fft(T_spectral)
%Ca=zeros(10,1)

dataqueue = parallel.pool.DataQueue;
afterEach(dataqueue, @fprintf);
fprintf('\nProgress of Phasor transformation:\n');
fprintf(repmat('.',1,img_height));
if usejava('desktop')
    fprintf('\r');
else
    fprintf(repmat('\b',1,imgHeight));
end





%% Discrete FT in matlab for count regi in camera

img_width=size(Data_regi, 2);
img_height=size(Data_regi,1);
band_num=size(Data_regi,3);
fre_num=band_num;

% using count after registration
Pd_regi=zeros(img_height,img_width, fre_num); % discrete ft phase
Pdn_regi=zeros(img_height,img_width, fre_num); % discrete ft phase
PAd_regi=zeros(img_height,img_width, fre_num);
PTd_regi=zeros(img_height,img_width, fre_num);
PUd_regi=zeros(img_height,img_width, fre_num); % real part
PVd_regi=zeros(img_height,img_width, fre_num); % img part

parfor i =1: img_height
    for j=1: img_width 
        Radiance=Data_regi(i,j,:);
        %Radiance= CRadiance(Ca); % Radiance 
        % % discrete ft of Ra
        % [Pu(i,j,:), Pv(i,j,:)]=phasor_cft(Ca,fre_num); % ft phasor
        % % continuals ft of RA average
        % [Pu2(i,j,:), Pv2(i,j,:)]=phasor_ft(Ca,fre_num); % ft phasor
        
         % discrete fft in matlab
        ARadiance=sum(Radiance);
        Pd_regi(i,j,:)=fft(Radiance);  % MATLAB
        Pdn_regi(i,j,:)=Pd_regi(i,j,:)/ARadiance;
        PAd_regi(i,j,:)=abs(Pd_regi(i,j,:)); % or sqrt(PUd.^2+PVd.^2) Euclidean distance
        PTd_regi(i,j,:)=angle(Pd_regi(i,j,:)); % angle theta
        PUd_regi(i,j,:)=real(Pd_regi(i,j,:));
        PVd_regi(i,j,:)=imag(Pd_regi(i,j,:));

        

    end 
    send(dataqueue,sprintf('|'));
end 


fprintf('\n Seperate round end! \n');

save  'Phasor_data/Pd_regi.mat' Pd_regi
save  'Phasor_data/Pdn_regi.mat' Pdn_regi
save  'Phasor_data/PUd_regi.mat' PUd_regi
save  'Phasor_data/PVd_regi.mat' PVd_regi
save  'Phasor_data/PTd_regi.mat' PTd_regi
save  'Phasor_data/PAd_regi.mat' PAd_regi
%%
duckim=adapthisteq(rescale(PAd_regi(:,:,1),0,1));
figure;
imshow(duckim);

