function [output] = CRadiance(Ca)

% Taking into account the camera response curve. 
%%%%Ca

%   Detailed explanation goes here
t1=20.0+273.15;
t2=20.0+273.15;
e1=0.1;
e2=0.59;

c=299792458; % m/s
h_bar=105457180e-42; % J/s
kb=138064852e-31; % J/k 
h=2*pi*h_bar;
            %obj.waveunit
            %obj.wav
cB=h*c/kb*1e6;
t=Ca(:)+273.15;
V=1;
emi_trial1=[0.95, 0.95, 0.95,0.95, 0.95,0.95, 0.95,0.95,0.95,0.95];

%S_EnvObj=load('S_EnvObj.mat').S_EnvObj;
%X=V*e1*BBp(t1)+(1-V)*e2*BBp(t2);
%BBp=obj.BBp(t)
%fun=emi_trial*BBp(t)+(1-emi_trial)*X
Sa=zeros(10,1);
Zz=[7.5,	14,	0.75; 7.5,	10.65,	0.672; 10.1,	11.5,	0.8487;
8.55,	14,	0.693; 10,	14,	0.63; 7.5,	11.234,	0.7783; 7.5,	8.67,	0.7106;
9.4,	14,	0.7093;  8.11,	14,	0.6993; 10,	11.23,	0.7052];  % the transmittance value and camera response are added, the response curve of the camera should also be added
% syms f(wav) [1-10]
% f1 = piecewise(wav<7.5, 0, wav>7.5&wav<14, 1, wav>14, 0)
% fplot(f1)

for i=1:10  
    %C(i) = integral(@(wav) Zz(i,3)*(emi_trial*BBp(wav, t)+(1-emi_trial)*V*e1*BBp(wav, t1)+(1-V)*e2*BBp(wav, t2)), Zz(i,1),Zz(i,2),'arrayvalued', true);
    Sa(i,1) = integral(@(wav) Zz(i,3)*emi_trial1(i)*((1e24.*(2*h*c^2)./wav.^5)./(exp(cB./(wav.*t(i)))-1)) ...
        , Zz(i,1), Zz(i,2));
    %Sa(i,1) = integral(@(wav) emi_trial1(i)*BBp(wav, t(i)), 7.5, 14); %*((1e24.*(2*h*c^2)./wav.^5)./(exp(cB./(wav.*t(i)))-1)) ...
       
        %*(V*e1*((1e24.*(2*h*c^2)./wav.^5)./(exp(cB./(wav.*t1))-1)) ...
        %+(1-V)*e2*((1e24.*(2*h*c^2)./wav.^5)./(exp(cB./(wav.*t2))-1)))), Zz(i,1), Zz(i,2)); %, 'ArrayValued', true
   %C(i,1)= integral(@(wav) (Zz(i,3)*(emi_trial*(1e14.*(2*h*c^2)./wav.^5)./(exp(cB./(wav.*t))-1))+(1-emi_trial)*V*S_EnvObj(i,1)+(1-V)*S_EnvObj(i,2)), Zz(i,1), Zz(i,2) );
end

%C
output = Sa; 

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