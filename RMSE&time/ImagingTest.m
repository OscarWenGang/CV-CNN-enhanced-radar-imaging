% test RMSEs and time needs of different algorihtms (FFT,SPGL1,RCNN,CCNN)

% Author£ºGao Jingkun  Date£º2017.12.19
clear;
%% parameters
e0=8.85e-12;u0=4*pi*1e-7;
c=1/sqrt(e0*u0);
N_pulse=1024*32;
N_fs=4096; 
fs=10e6;
T=500e-6;
Tp=400e-6; 
f_start=13449*1e6;
f_end=14250*1e6;
B=(f_end-f_start)*16;
N_f=fs*Tp;
f=linspace(f_start*16,f_end*16,N_f);
index_f_used=1:8:4000;
f=f(index_f_used);f=f(:);
N_f=length(f);
rotatespeed=90/4;
fai=linspace(0,rotatespeed*T*(N_pulse-1),N_pulse);
angle_range=360;
N_fai=floor(angle_range/fai(end)*N_pulse);
fai=fai(1:N_fai);fai=fai(:);
d_f=f(2)-f(1); 
d_fai=fai(2)-fai(1);
range_r=c/2/d_f;  
range_a=c/mean(f)/2/deg2rad(d_fai);
d_x_Ima=0.003;
d_y_Ima=d_x_Ima; 
Nfft_r=round(range_r/d_x_Ima);
Nfft_a=round(range_a/d_y_Ima);
N_x=280;
N_y=280;
x=linspace(0,d_x_Ima*N_x,N_x+1);x=x(:);
x=[x(ceil(N_x/2+0.5)+mod(N_x,2):end-1)-x(end);x(1:ceil(N_x/2))];
y=linspace(0,d_y_Ima*N_y,N_y+1);y=y(:);
y=[y(ceil(N_y/2+0.5)+mod(N_y,2):end-1)-y(end);y(1:ceil(N_y/2))];
[X,Y]=ndgrid(x,y);
N_fai=300;
fai=linspace(0,d_fai*N_fai,N_fai+1);fai=fai(:);
fai=[fai(ceil(N_fai/2+0.5)+mod(N_fai,2):end-1)-fai(end);fai(1:ceil(N_fai/2))];
I=eye(2);
N_FFT=[Nfft_r,Nfft_a];
N_sig=[N_f,N_fai];
N_Ima=[N_x,N_y];
%% importdata % setup
Sig=importdata('Sig.mat');
Output=importdata('Output.mat');
Implicit_A = @(s,mode) Impl_A_At(s,N_FFT,N_sig,N_Ima,mode);
addpath('..\dependence\spgl1-1.9');
addpath('..\Train_CCNN');
addpath('..\Train_RCNN');
opts = spgSetParms('verbosity',0);
load('..\Train_CCNN\CCNN_net.mat','netNet','netTopo'); 
netNet_CCNN=netNet;
netTopo_CCNN=netTopo;
load('..\Train_RCNN\RCNN_net.mat','netNet','netTopo');
netNet_RCNN=netNet;
netTopo_RCNN=netTopo;
clear netNet netTopo;
%% add noise
N_sample=100;
SNR=0;                      % SNR
for c_sample=1:N_sample
    Sig(:,:,c_sample)=awgn(Sig(:,:,c_sample),SNR,'measured');
end
%% test
T_FFT=zeros(N_sample,1);
T_SPGL1=T_FFT;
T_RCNN=T_FFT;
T_CCNN=T_FFT;
RMSE_FFT=zeros(N_sample,1);
RMSE_SPGL1=RMSE_FFT;
RMSE_RCNN=RMSE_FFT;
RMSE_CCNN=RMSE_FFT;
for c_sample=1:N_sample
% FFT
sig=Sig(:,:,c_sample);
tic;
c_t=1;
N_pad_pre=ceil((N_FFT(c_t)-N_sig(c_t))/2);
N_pad_post=N_FFT(c_t)-N_sig(c_t)-N_pad_pre;
sig=padarray(sig,N_pad_pre*I(c_t,:),0,'pre');
sig=padarray(sig,N_pad_post*I(c_t,:),0,'post');
sig=ifft(ifftshift(sig,c_t),[],c_t);
sig=[sig(end-N_Ima(1)+ceil(N_Ima(1)/2)+1:end,:);sig(1:ceil(N_Ima(1)/2),:)];
c_t=2;
N_pad_pre=ceil((N_FFT(c_t)-N_sig(c_t))/2);
N_pad_post=N_FFT(c_t)-N_sig(c_t)-N_pad_pre;
sig=padarray(sig,N_pad_pre*I(c_t,:),0,'pre');
sig=padarray(sig,N_pad_post*I(c_t,:),0,'post');
sig=ifft(ifftshift(sig,c_t),[],c_t);
sig=[sig(:,end-N_Ima(2)+ceil(N_Ima(2)/2)+1:end) sig(:,1:ceil(N_Ima(2)/2))];
t_FFT=toc;

% SPGL1
ss=Sig(:,:,c_sample);
sigma=sqrt(norm(ss(:),2)^2/(db2pow(SNR)+1));
tic;
Ima = spg_bpdn(Implicit_A,ss(:),sigma,opts);
t_SPGL1=toc;
Ima=reshape(abs(Ima),N_Ima);

% RCNN
tic;
inPut(:,:,1)=gpuArray(real(sig));
inPut(:,:,2)=gpuArray(imag(sig));
[ netFture_RCNN, ~ ] = JK_feedForward_nn_r( netNet_RCNN, netTopo_RCNN, 9*single(inPut));
t_RCNN=toc;

% CCNN
tic;
[ netFture_CCNN, ~ ] = JK_feedForward_nn( netNet_CCNN, netTopo_CCNN, gpuArray(complex(9*single(sig))));
t_CCNN=toc;

% record
outS=size(netFture_CCNN{end, 1});
shrinkSx=N_x-outS(1);
shrinkSy=N_y-outS(2);
outPut=Output(floor(shrinkSx/2)+1:end-ceil(shrinkSx/2),floor(shrinkSy/2)+1:end-ceil(shrinkSy/2),c_sample);
Ima_FFT=abs(sig(floor(shrinkSx/2)+1:end-ceil(shrinkSx/2),floor(shrinkSy/2)+1:end-ceil(shrinkSy/2)))*(Nfft_r*Nfft_a)/(N_f*N_fai);
Ima_SPGL1=Ima(floor(shrinkSx/2):end-ceil(shrinkSx/2)-1,floor(shrinkSy/2):end-ceil(shrinkSy/2)-1)*sqrt(Nfft_r*Nfft_a)/(N_f*N_fai); % /5;
Ima_RCNN=gather(abs(netFture_RCNN{end, 1}(:,:,1,1)));
Ima_CCNN=gather(abs(netFture_CCNN{end, 1}(:,:,1,1)));

xxx=Ima_SPGL1(:);yyy=outPut(:);
aaa=xxx'*yyy/(xxx'*xxx);
T_FFT(c_sample)=t_FFT;
T_SPGL1(c_sample)=t_SPGL1;
T_RCNN(c_sample)=t_FFT+t_RCNN;
T_CCNN(c_sample)=t_FFT+t_CCNN;

RMSE_FFT(c_sample)=sqrt(sum(sum((Ima_FFT-outPut).^2,1),2)/outS(1)/outS(2));
RMSE_SPGL1(c_sample)=sqrt(sum(sum((aaa*Ima_SPGL1-outPut).^2,1),2)/outS(1)/outS(2));
RMSE_RCNN(c_sample)=sqrt(sum(sum((Ima_RCNN-outPut).^2,1),2)/outS(1)/outS(2));
RMSE_CCNN(c_sample)=sqrt(sum(sum((Ima_CCNN-outPut).^2,1),2)/outS(1)/outS(2));
end

save 0dB SNR T_FFT T_SPGL1 T_RCNN T_CCNN RMSE_FFT RMSE_SPGL1 RMSE_RCNN RMSE_CCNN


