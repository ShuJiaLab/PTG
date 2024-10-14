%%% compile the hdr file for hyperspectral data

 dimensions =[480,640,10]; 
 myFilename='facecount.bsq';
 precisionOfData='double';
 readOffset =0;
 interleaveType = 'bsq';
 byteOrdering = 'ieee-le';

 for num=1:10
    hc = importdata(['Motor_corey_temperature/corey10_',num2str(num),'.mat']);
    data(:, :, num)=hc.Frame;
    %figure;
    %imshow(hc.Frame);
    %adapthisteq(rescale(obj.xMap,0,1));
end
face=data;

for num=1:10
    hc = importdata(['Motor_corey_count/corey10_',num2str(num),'.mat']);
    data_count(:, :, num)=hc.Frame;
    %figure;
    %imshow(hc.Frame);
    %adapthisteq(rescale(obj.xMap,0,1));
end
face_count=data_count;

% for num=1:10
%     hc = importdata(['Corey_nolight_noglasses_radiance/corey_',num2str(num),'.mat']);
%     data_ra(:, :, num)=hc.Frame;
%     %figure;
%     %imshow(hc.Frame);
%     %adapthisteq(rescale(obj.xMap,0,1));
% end
% face_ra=data_ra;


multibandwrite(face_count, myFilename ,'bsq');
myFilename='face.bsq';
multibandwrite(face, myFilename ,'bsq');

%% data load from bsq
newData = multibandread('facecount.bsq',dimensions,precisionOfData,readOffset,interleaveType,byteOrdering);
%multibandread(myFilename, dimensions, precisionOfData, readOffset, interleaveType, byteOrdering);
%% test output
handim=adapthisteq(rescale(newData(:,:,10),0,1));
figure;
imshow(handim);