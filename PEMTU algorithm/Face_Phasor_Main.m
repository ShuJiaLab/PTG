% *************************************************************************
%    Phasor Thermography: Phasor-Enabled Multiparametric Thermal  Unmixing
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
% loading data

c=299792458; % m/s
h_bar=105457180e-42; % J/s
kb=138064852e-31; % J/k 
h=2*pi*h_bar;
CB=h*c/kb*1e6;
CC=1e30*2*h*c^2;
data=rand(480, 640, 10);
emap=zeros(480,640);
vmap=zeros(480,640);
tmap=zeros(480,640);

b=randn(3,1);
y=randn(10,1);
% FILTER INFO
Filter=[7.5, 14, 1; 8, 10.65, 0.9448; 10.1, 11.5, 0.8315;8.55, 14,0.9036; 10.0, 14, 0.9156; 7.5, 11.234, 0.944; 7.5, 8.67, 0.957; 9.4, 14, 0.9309; 8.11, 14, 0.9157; 10, 11.23, 0.836 ]  % the transmittance value is also added, the response curve of the camera should also be added

x=randn(3);
upperlimit=14;
lowerlimit=7.4;
%% Camera responce 
Rca=[0.75; 0.84; 0.943;0.77;0.7;0.846; 0.748;0.771;0.777;0.86 ]
Zz= Filter;
Zz(:,3)=Filter(:,3); % input for integration; 

% % mirror= importdata(['Head_sculpture\Mirror.mat']);
% % r_m(:, :)=10000*mirror.Frame;

save 'Zz.mat' Zz ;
% for num=1:10
%     hc = importdata(['Motor_corey_temperature/corey10_',num2str(num),'.mat']);
%     data(:, :, num)=hc.Frame;  % unit change to watt/m^2.sr
%     %data(:, :, num)=hc.Frame;  % unit count
%     %figure;
%     %imshow(hc.Frame);
%     %adapthisteq(rescale(obj.xMap,0,1));
% end
dimensions =[451,481,10];
myFilename='motorcoreytemp_regi.bsq';
precisionOfData='double';
readOffset =0;
interleaveType = 'bsq';
byteOrdering = 'ieee-le';

Datatemp_regi = multibandread('motorcoreytemp_regi.bsq',dimensions,precisionOfData,readOffset,interleaveType,byteOrdering);
save  'Datatemp_regi.mat' Datatemp_regi
%changing the value based on the unit if needed
 %unitfactor=0.01;
head=Datatemp_regi;
%matLib=[0.88,; 0.92,; 0.95,;   0.97; 0.98];
% matLib=[
%      0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88;
%     0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94;
%     0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96,0.96, 0.96; 
%     0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95;
%     0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91,0.91, 0.91;
% ];

matLib=[
      0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88;
     0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94;
     0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94,0.94, 0.94; 
     0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95;
     0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91,0.91, 0.91;
 ];
matLib=matLib';
save 'matLib.mat' matLib;
hueLib=[0.039215686274510, 0.117647058823529, 0.7732,0.9132,  0.3250];
save 'hueLib.mat' hueLib; 
numBeta=2;
numMat=5;
S_EnvObj=load('S_EnvObj.mat').S_EnvObj; %[50.3799000000000	20.2438000000000	10.3649000000000	38.2650000000000	27.4579000000000	27.6844000000000	8.38400000000000	32.7013000000000	41.3550000000000	8.35450000000000; 31.6214000000000	12.7062000000000	6.50570000000000	24.0174000000000	17.2342000000000	17.3764000000000	5.26230000000000	20.5253000000000	25.9569000000000	5.24380000000000]
emap_phasor=load('emap2_4.mat').eMapimg2_4;

% 
[imgHeight,imgWidth] = size(head(:,:,1));
t0 = zeros(imgHeight,imgWidth);
%tMap0 = zeros(imgHeight,imgWidth)+15;
%v0 = zeros(imgHeight,imgWidth);
vmap_phasor=load('md_d2_4.mat').md_d2_4;
% 
%[imgHeight,imgWidth] = size(head(:,:,1));
%t0 = zeros(imgHeight,imgWidth);
%tMap0 = zeros(imgHeight,imgWidth)+15;
%v0 = zeros(imgHeight,imgWidth);
v0 = rescale(vmap_phasor, 0,1); % k-means clustering
figure;
imshow(adapthisteq(v0),[])
title('Texture map from phasor')
%hct= importdata(['Motor_corey_temperature/corey10_1.mat']);
t0(:, :)= zeros(imgHeight,imgWidth)+20; %head(:,:,1);
rawface=head(:,:,1);
figure;imshow(rawface,[]);%imshow(adapthisteq(rescale(head(:,:,1),0,1)));
title('Thermal image with no filter')
% mirror= importdata(['Head_sculpture\Mirror.mat']);
% t_m(:, :)=mirror.Frame;
% t0_scale=t0(:,:)-t_m(:,:);
% figure;imshow(hct.Frame,[]);
% title('Thermal image with mirror scaling')
%% 
%t0=data(:,:,1) ;% set the initial value to be the state of the art thermal temperature value
%v0= zeros(imgHeight,imgWidth);
%Aeq = [0,   ones(1,numBeta).*boolean(sum(S_EnvObj,2)')]
%Aeq=[0, ones(1,numBeta).*boolean(sum(S_EnvObj,2))'];



Aeq=[0, 1 ];
Beq = 1;
eMap0 = zeros(imgHeight,imgWidth)+NaN;
tMap0 = zeros(imgHeight,imgWidth)+NaN;
vMap0 = v0;
resMap0 = zeros(imgHeight,imgWidth)+NaN;

%OPT=optimset('Algorithm','interior-point','MaxIter',1e3,'display','off');
%OPT = optimoptions('MaxIter',1e3, 'display','off' );
% algorithms: levenberg-marquardt

dataqueue = parallel.pool.DataQueue;
afterEach(dataqueue, @fprintf);
fprintf('\nProgress of nonlinear regression decomposition:\n');
fprintf(repmat('.',1,imgHeight));
if usejava('desktop')
    fprintf('\r');
else
    fprintf(repmat('\b',1,imgHeight));
end
parfor indl=1:imgHeight
        cost = zeros(1,1);
        paraList = zeros(numBeta);  % TWO PARAMETERS TO rows
        
        for inds=1:imgWidth
            Ca = squeeze(head(indl,inds,:));
            Sa=CRadiance(Ca);
            V=vmap_phasor(indl,inds);
            %pause(inf)
%if obj.parallelComputing 
            %for indt=1:numMat
                indt=emap_phasor(indl,inds); % eMap phasor
                emi_trial = matLib(:, indt);  % here 
                fitParameter=[t0(indl,inds)]%[t0(indl,inds);squeeze(v0(indl,inds))]; % initial guess
                LB=[t0(indl,inds)-20; zeros(1,1)];
                UB=[t0(indl,inds)+20;  ones(1,1)];
                %[fitParameter,resnorm]=lsqnonlin(@(x) Cost_Ca(emi_trial,Ca,x),fitParameter0,LB,UB, [], [], Aeq, Beq, [], OPT );%,[],[],Aeq,Beq,LB,UB,[],OPT);
                [fitParameter,resnorm]=lsqnonlin(@(x) Cost_Ca(emi_trial,Sa,V,x),fitParameter,LB,UB);
                cost=resnorm;
                paraList=fitParameter;
            %end
            %[resMap0(indl,inds),indt2]=min(cost);   %% !!!!
            resMap0(indl,inds)=cost;
            %fitParameter=paraList(:,indt2);
            eMap0(indl,inds) = indt;  %%% material assignment
            tMap0(indl,inds) = fitParameter(1);
            %vMap0(indl,inds) = ;
        end
        send(dataqueue,sprintf('|'));
end
fprintf('\n\n');



%% iteration with fixed eMap
eMap=eMap0;
save 'eMap0.mat' eMap0;
save 'vMap0.mat' vMap0;
save 'tMap0.mat' tMap0;
save 'resMap.mat' resMap0;

%hct= importdata(['Corey_glasses/Corey_1.mat']);
%t0(:, :)=hct.Frame;

% eMap1 = zeros(imgHeight,imgWidth)+NaN;
% tMap1 = zeros(imgHeight,imgWidth)+NaN;
% vMap1 = zeros(imgHeight,imgWidth)+NaN;
% resMap1 = zeros(imgHeight,imgWidth)+NaN;
% 
% for indl=1:imgHeight
%         cost2 = 0;
%         paraList1 = zeros(numBeta,1);  % TWO PARAMETERS TO rows
%         for inds=1:imgWidth
%             Ca = squeeze(hand(indl,inds,:));
%             Sa=CRadiance(Ca);
%             %pause(inf)
% %if obj.parallelComputing 
%             %for indt=1:numMat
%                 indt=eMap0(indl,inds);
%                 emi_trial = matLib(:, indt);  % here 
%                 fitParameter=[t0(indl,inds);squeeze(v0(indl,inds))]; % initial guess
%                 LB=[t0(indl,inds)-5; zeros(1,1)];
%                 UB=[t0(indl,inds)+5;  ones(1,1)];
%                 %[fitParameter,resnorm]=lsqnonlin(@(x) Cost_Ca(emi_trial,Ca,x),fitParameter0,LB,UB, [], [], Aeq, Beq, [], OPT );%,[],[],Aeq,Beq,LB,UB,[],OPT);
%                 [fitParameter,resnorm]=lsqnonlin(@(x) Cost_Ca(emi_trial,Sa,x),fitParameter,LB,UB);
%                 cost2=resnorm;
%                 paraList(:,indt)=fitParameter;
%             %end
%             resMap1(indl,inds)=cost2;   %% !!!!
%             fitParameter=paraList(:);
%             %eMap0(indl,inds) = indt2;  %%% material assignment
%             tMap1(indl,inds) = fitParameter(1);
%             vMap1(indl,inds) = fitParameter(2);
%         end
%         %send(dataqueue,sprintf('|'));
% end

tMap=tMap0;
vMap=vMap0;
resMap=resMap0;

%%% finalization of TeX
%% X map from V to X  do the integration 

%xMap=vMap.*BBx(eMap,t1)+(1-vMap).*e2.*BBx(eMap, t2);
t1=20.5;
t2=20.5;
e1=0.7;
e2=0.59;

xMap=zeros(imgHeight, imgWidth);
%xMap=Xmap_calc(vMap);
% for i=1:imgHeight
% for j =1: imgWidth
%     xMap(i,j)=vMap(i,j,1)*BBxx(e1, t1)+vMap(i,j,2)*BBxx(e2, t2);
% 
% end
% end
% 
% xMap_b = adapthisteq(rescale(xMap,0,1));


%%

save 'tMap.mat' tMap
save 'vMap.mat' vMap

save 'resMap.mat' resMap

%save 
save(['face_TeX_Data/','eMap','.mat'],'-mat','eMap');
save(['face_TeX_Data/','tMap','.mat'],'-mat','tMap');

save(['face_TeX_Data/','resMap','.mat'],'-mat','resMap');
%save(['hand_TeX_Data/','resMap',num2str(obj.iter),'.mat'],'-mat','resMap');



%%%%%% Visulization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tqMin =0.006;
tqMax =0.995; 
%'xAGC','nonlinear',...
xqMin=0.03;
xqMax=0.97;
stripeThresh=3;
noiseThresh=3;
%'plot',true);

% Hue
[imgHeight,imgWidth] = size(eMap);
eMap_h = reshape(hueLib(reshape(eMap,[],1)),[imgHeight,imgWidth]);
save 'eMap_h.mat' eMap_h
% Saturation
tMap_s = rescale(tMap,0,1,'InputMin',quantile(reshape(tMap,[],1),tqMin),'InputMax',quantile(reshape(tMap,[],1),tqMax));
% Brightness
%if strcmp(options.xAGC,'nonlinear')
    %xMap = adapthisteq(rescale(xMap,0,1));
%else
%    xMap = rescale(obj.xMap,0,1,'InputMin',quantile(reshape(obj.xMap,[],1),options.xqMin),'InputMax',quantile(reshape(obj.xMap,[],1),options.xqMax));
%end
% denoiser
%%[xMap_v,~] = obj.denoiser(xMap,noiseThresh);

%% get the xMap from v map and resmap

%xMap=vMap+resMap;

% for i=1:imgHeight
% for j =1: imgWidth
%     xMap(i,j)=vMap(i,j)*BBxx(e1, t1)+(1-vMap(i,j))*BBxx(e2, t2);
% 
% end
% end
xMap=Xmap_calc(vMap);
save 'xMap.mat' xMap
save(['face_TeX_Data/','xMap','.mat'],'-mat','xMap');
xMap_b = adapthisteq(rescale(xMap,0,1));
figure;
imshow(xMap_b,[]);
title('XMap');


%% Visualization
TeXVision_hsv = cat(3,eMap_h,tMap_s,xMap_b);
TeXVision_rgb = hsv2rgb(TeXVision_hsv);
%if options.plot
     figure;image(TeXVision_rgb);axis off;axis image;
%end
title('TeX vision');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% least squares linear regression

%syms lmd;
%tau=[piecewise(lmd<14, 1, lmd)]
%x(1,:)=xx(k,:);

% numerical integral
% 


% b=[1, 2, 3]
% b*3
% radiance = integral_bbr(b,xx)

%bbr=@(wav,T) CC./(wav.^5)./(exp(CB./wav*T)-1);
%bbr((1:10), 1)
wav=[10.75, 9.075, 10.8, 11.275, 12, 9.367, 8.085, 11.7, 11.055, 10.675];% CENTER OF THE WAVE BAND

% for i=1:480
%     for j=1:640
%        % for k=1:2
%         y(:)=data(i,j,:); % response variable
%         bbr=@(wav,T) CC./(wav.^5)./(exp(CB./wav.*T)-1); 
%         funb=@(wav, b) b(1)*bbr(wav,b(3))+(1-b(1))*(b(2)*0.94*bbr(wav,293)+(1-b(2))*0.59*bbr(wav,293)); % transmitance function should be added here 
% 
%         %x(1,:)=xx(k,:);
%         %x(:)=xx(k,:)
%         %intgralfun=@(wav, b,x1, x2) integral(@(wav) xx(:,3).*funb(wav,b), x1, x2);
%         %modelfun=@(b,x) x(:,3).*b(1).*(CC./(((x(:,1)+x(:,2))/2).^5./(exp(CB./(x(:,1)+x(:,2))/2).*x(:,3)-1)))
%         modelfun=@(b,x) x(:,3).*(x(:,2)-x(:,1)).*(b(1)*bbr((x(:,1)+x(:,2)/2),b(3))+(1-b(1))*((b(2)*0.94)*bbr((x(:,1)+x(:,2)/2),293)+(1-b(2)*0.59*bbr((x(:,1)+x(:,2)/2), 293))))
%         %modelfun=@(b,xx) integral(@(wav) xx(:,3).*funb(wav,b), xx(:,1), xx(:,2));
%         %modelfun=@(b,x) integral(@(wav) x(1,3)*b(1)*(CC/(wav.^5)*(1/(exp(CB/wav*b(3))-1)))+(1-b(1))*(b(2)*0.94*(CC/(wav.^5)*(1/(exp(CB/wav*293.15)-1)))+(1-b(2))*0.59*(CC/(wav.^5)*(1/(exp(CB/wav*293.15)-1))))), x(1,3), x(1,2))
%         %modelfun=@(b,x) arrayfun(@(x_(:,1), x_(:,2), x_(:,3)) integral(@(wav) x_(:,3).*funb(wav,b), x_(:,1), x_(:,2)), x);
%         %modelfun=@(b,x ) x(:,3).*integral(@(wav) (b(1)*(CC/(wav.^5)*(1/(exp(CB/wav.*b(3))-1)))+(1-b(1))*(b(2)*0.94*(CC/(wav.^5)*(1/(exp(CB/wav.*293.15)-1)))+(1-b(2))*0.59*(CC/(wav.^5)*(1/(exp(CB/wav.*293.15)-1))))), x(:,1), x(:,2));
%         md1=nlinfit(xx, y, modelfun, b);
%         emap(i,j)=md1(1);
%         vmap(i,j)=md1(2);
%         tmap(i,j)=md1(3);
% 
% 
%         %end
%     end
% end
%         y(:)=data(i,j,:); % response variable
%         bbr=@(wav,T) CC./(wav.^5)./(exp(CB./wav.*T)-1); 
%         funb=@(wav, b) b(1)*bbr(wav,b(3))+(1-b(1))*(b(2)*0.94*bbr(wav,293)+(1-b(2))*0.59*bbr(wav,293)); % transmitance function should be added here 
% 
%         %x(1,:)=xx(k,:);
%         %x(:)=xx(k,:)
%         %intgralfun=@(wav, b,x1, x2) integral(@(wav) xx(:,3).*funb(wav,b), x1, x2);
%         %modelfun=@(b,x) x(:,3).*b(1).*(CC./(((x(:,1)+x(:,2))/2).^5./(exp(CB./(x(:,1)+x(:,2))/2).*x(:,3)-1)))
%         modelfun=@(b,x) x(:,3).*(x(:,2)-x(:,1)).*(b(1)*bbr((x(:,1)+x(:,2)/2),b(3))+(1-b(1))*((b(2)*0.94)*bbr((x(:,1)+x(:,2)/2),293)+(1-b(2)*0.59*bbr((x(:,1)+x(:,2)/2), 293))))
%         %modelfun=@(b,xx) integral(@(wav) xx(:,3).*funb(wav,b), xx(:,1), xx(:,2));
%         %modelfun=@(b,x) integral(@(wav) x(1,3)*b(1)*(CC/(wav.^5)*(1/(exp(CB/wav*b(3))-1)))+(1-b(1))*(b(2)*0.94*(CC/(wav.^5)*(1/(exp(CB/wav*293.15)-1)))+(1-b(2))*0.59*(CC/(wav.^5)*(1/(exp(CB/wav*293.15)-1))))), x(1,3), x(1,2))
%         %modelfun=@(b,x) arrayfun(@(x_(:,1), x_(:,2), x_(:,3)) integral(@(wav) x_(:,3).*funb(wav,b), x_(:,1), x_(:,2)), x);
%         %modelfun=@(b,x ) x(:,3).*integral(@(wav) (b(1)*(CC/(wav.^5)*(1/(exp(CB/wav.*b(3))-1)))+(1-b(1))*(b(2)*0.94*(CC/(wav.^5)*(1/(exp(CB/wav.*293.15)-1)))+(1-b(2))*0.59*(CC/(wav.^5)*(1/(exp(CB/wav.*293.15)-1))))), x(:,1), x(:,2));
%         md1=nlinfit(xx, y, modelfun, b);

%plotSlice(mdl);
% function radiance = integral_bbr(b,x)
%     c=299792458; % m/s
%     h_bar=105457180e-42; % J/s
%     kb=138064852e-31; % J/k 
%     h=2*pi*h_bar;
%     CB=h*c/kb*1e6;
%     CC=1e30*2*h*c^2;
% 
%     num=10
%     length=size(x,1)
%     wav=zeros(length,num)
%     dt=zeros(length,1)
%     dt(:)=(x(:,2)-x(:,1))./num
%     %wav(:,1)=x(:,1)+dt(:)
%     e1=0.94;
%     e2=0.59;
%     heatC=zeros(size(x,1))
%     funb=zeros(size(x,1))
%     for i=1:num
%         wav(:,i)=x(:,1)+dt(:)
%         funb(:)=x(:,3).*b(1).*(CC/(wav(:,i).^5)./(exp(CB/wav(:,i).*b(3))-1))+(1-b(1))*(b(2)*0.94*(CC/(wav(:,i).^5)/(exp(CB/wav(:,i).*293.5)-1))+(1-b(2))*0.59*(CC/(wav(:,i).^5)./(exp(CB/wav(:,i).*293.5)-1)))
%         heatC(:)=heatC(:)+funb(:)
%     end
%     radiance=heatC
% end
function Sradiance = BBp(wav,te)
         c=299792458; % m/s
         h_bar=105457180e-42; % J/s
         kb=138064852e-31; % J/k 
         h=2*pi*h_bar;
            %obj.waveunit
            %obj.wav
         cB=h*c/kb*1e6;
         Sradiance = (1e24.*(2*h*c^2)./wav.^5)./(exp(cB./(wav.*te))-1); % blackbody radiation power spectrum
         %radiance
end
% Radiance with no filter
function Radiancet = BBrt(emi_trial,te)

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
t1=20.5+273.15;
t2=20.5+273.15;
e1=0.28;
e2=0.59;
t=te+273.15;
%V=input(2);
%X=V*e1*BBp(t1)+(1-V)*e2*BBp(t2);
%BBp=obj.BBp(t)
%fun=emi_trial*BBp(t)+(1-emi_trial)*X
C=zeros(10,1)
%Zz=[7.5, 14, 1; 7.5, 10.65, 0.8; 10.1, 11.5, 0.9;8.55, 14,0.9; 10.0, 14, 0.9; 7.5, 11.234, 0.92; 7.5, 8.67, 0.95; 9.4, 14, 0.92; 8.11, 14, 0.9; 10, 11.23, 0.82 ]  % the transmittance value is also added, the response curve of the camera should also be added
for i=1:10
    C(i,1) = integral(@(wav) emi_trial*BBp(wav, t), 7.5, 14);
end
Radiancet=C;

end

%  bbr with filters

function Radiance = BBr(emi_trial,te)

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
t1=20.0+273.15;
t2=20.0+273.15;
e1=0.28;
e2=0.59;
t=te+273.15;
V=1
%V=input(2);
%X=V*e1*BBp(t1)+(1-V)*e2*BBp(t2);
%BBp=obj.BBp(t)
%fun=emi_trial*BBp(t)+(1-emi_trial)*X
C=zeros(10,1)
Zz=[7.5, 14, 1; 7.5, 10.65, 0.8; 10.1, 11.5, 0.9;8.55, 14,0.9; 10.0, 14, 0.9; 7.5, 11.234, 0.92; 7.5, 8.67, 0.95; 9.4, 14, 0.92; 8.11, 14, 0.9; 10, 11.23, 0.82 ]  % the transmittance value is also added, the response curve of the camera should also be added
for i=1:10
    C(i,1) = integral(@(wav) Zz(i,3)*(emi_trial*BBp(wav, t)+(1-emi_trial)*(V*e1*BBp(wav, t1)+(1-V)*e2*BBp(wav, t2))), Zz(i,1), Zz(i,2));
end
Radiance=C;

end

%%%%Environment Radiance %%%%%%%%%%%%%%%%%

function FERadiance = BBx(emi,te)

%UNTITLED Summary of this function goes here
%   Radiance with fileter added to get the environment radiance from the
%   people
t1=20.0+273.15;
t2=20.0+273.15;
e1=0.28;
e2=0.59;
t=te+273.15;
V1=0.5;
V2=0.5;
%V=input(2);
%X=V*e1*BBp(t1)+(1-V)*e2*BBp(t2);
%BBp=obj.BBp(t)
%fun=emi_trial*BBp(t)+(1-emi_trial)*X
C=zeros(10,1);
Zz=[7.5, 14, 1; 7.5, 10.65, 0.8; 10.1, 11.5, 0.9;8.55, 14,0.9; 10.0, 14, 0.9; 7.5, 11.234, 0.92; 7.5, 8.67, 0.95; 9.4, 14, 0.92; 8.11, 14, 0.9; 10, 11.23, 0.82 ]  % the transmittance value is also added, the response curve of the camera should also be added
for i=1:10
    C(i,1) = integral(@(wav) Zz(i,3)*(emi*BBp(wav, t)+(1-emi)*(V1*e1*BBp(wav, t1)+V2*e2*BBp(wav, t2))), Zz(i,1), Zz(i,2));
end
FERadiance=C;

end


function ERadiance = BBxx(emi,te)

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
t1=20.5+273.15;
t2=20.5+273.15;
e1=0.28;
e2=0.59;
t=te+273.15;
V1=0.025;
V2=0.025;
%V=input(2);
%X=V*e1*BBp(t1)+(1-V)*e2*BBp(t2);
%BBp=obj.BBp(t)
%fun=emi_trial*BBp(t)+(1-emi_trial)*X
%C=zeros(10,1);
%Zz=[7.5, 14, 1; 7.5, 10.65, 0.8; 10.1, 11.5, 0.9;8.55, 14,0.9; 10.0, 14, 0.9; 7.5, 11.234, 0.92; 7.5, 8.67, 0.95; 9.4, 14, 0.92; 8.11, 14, 0.9; 10, 11.23, 0.82 ]  % the transmittance value is also added, the response curve of the camera should also be added
%for i=1:10
C= integral(@(wav) emi*BBp(wav, t)+(1-emi)*(V1*e1*BBp(wav, t1)+V2*e2*BBp(wav, t2)), 7.5, 14);
%end
ERadiance=C;

end

% loss function
function [output] = CostC(emi_trial,Ca,input)

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
t1=20.5+273.15;
t2=20.5+273.15;
e1=0.28;
e2=0.59;

c=299792458; % m/s
h_bar=105457180e-42; % J/s
kb=138064852e-31; % J/k 
h=2*pi*h_bar;
            %obj.waveunit
            %obj.wav
cB=h*c/kb*1e6;
t=input(1)+273.15;
V=input(2);
%X=V*e1*BBp(t1)+(1-V)*e2*BBp(t2);
%BBp=obj.BBp(t)
%fun=emi_trial*BBp(t)+(1-emi_trial)*X
C=zeros(10,1);
Zz=[7.5, 14, 1; 
    7.5, 10.65, 0.8; 
    10.1, 11.5, 0.9;
    8.55, 14,0.9;
    10.0, 14, 0.9; 
    7.5, 11.234, 0.92;
    7.5, 8.67, 0.95; 
    9.4, 14, 0.92; 
    8.11, 14, 0.9; 
    10, 11.23, 0.82 ];  % the transmittance value is also added, the response curve of the camera should also be added
% syms f(wav) [1-10]
% f1 = piecewise(wav<7.5, 0, wav>7.5&wav<14, 1, wav>14, 0)
% fplot(f1)

for i=1:10  
    %C(i) = integral(@(wav) Zz(i,3)*(emi_trial*BBp(wav, t)+(1-emi_trial)*V*e1*BBp(wav, t1)+(1-V)*e2*BBp(wav, t2)), Zz(i,1),Zz(i,2),'arrayvalued', true);
    C(i,1) = integral(@(wav) Zz(i,3)*(emi_trial*((1e24.*(2*h*c^2)./wav.^5)./(exp(cB./(wav.*t))-1)) ...
        +(1-emi_trial) ...
        *(V*e1*((1e24.*(2*h*c^2)./wav.^5)./(exp(cB./(wav.*t1))-1)) ...
        +(1-V)*e2*((1e24.*(2*h*c^2)./wav.^5)./(exp(cB./(wav.*t2))-1)))), Zz(i,1), Zz(i,2)); %, 'ArrayValued', true
end
output = Ca - C; %vector? 
end
