% Generate training data
% Author£ºGao Jingkun  Date£º2017.12.04

%% parameters
% clear;
e0=8.85e-12;u0=4*pi*1e-7;
c=1/sqrt(e0*u0);                             % speed of light
N_pulse=1024*32;
N_fs=4096;
fs=10e6;
T=500e-6;
Tp=400e-6;
f_start=13449*1e6;                           
f_end=14250*1e6;
B=(f_end-f_start)*16;                        % bandwidth
N_f=fs*Tp;
f=linspace(f_start*16,f_end*16,N_f);
index_f_used=1:8:4000;                       % number of frequency samplings: 500
f=f(index_f_used);f=f(:);
N_f=length(f);
rotatespeed=90/4;                            % rotation speed of the turntable
fai=linspace(0,rotatespeed*T*(N_pulse-1),N_pulse);
angle_range=360;
N_fai=floor(angle_range/fai(end)*N_pulse);
fai=fai(1:N_fai);fai=fai(:);
d_f=f(2)-f(1);                               % ferquency interval
d_fai=fai(2)-fai(1);                         % azimuth angle interval
N_fai=300;                                   % number of azimuth angle samplings: 300
fai=linspace(0,d_fai*N_fai,N_fai+1);fai=fai(:);
fai=[fai(ceil(N_fai/2+0.5)+mod(N_fai,2):end-1)-fai(end);fai(1:ceil(N_fai/2))];
range_r=c/2/d_f;
range_a=c/mean(f)/2/deg2rad(d_fai);
d_x_Ima=0.003;                               % pixel interval
d_y_Ima=d_x_Ima;
Nfft_r=round(range_r/d_x_Ima);
Nfft_a=round(range_a/d_y_Ima);
N_x=180;                                     % pixel numbers of the groundtruth image for training
N_y=180;
x=linspace(0,d_x_Ima*N_x,N_x+1);x=x(:);
x=[x(ceil(N_x/2+0.5)+mod(N_x,2):end-1)-x(end);x(1:ceil(N_x/2))];
y=linspace(0,d_y_Ima*N_y,N_y+1);y=y(:);
y=[y(ceil(N_y/2+0.5)+mod(N_y,2):end-1)-y(end);y(1:ceil(N_y/2))];
[X,Y]=ndgrid(x,y);
%% Generate the echo signal and the groundtruth image
N_t_max=200;
k_tensor=2*pi*f/c*ones(1,N_fai);
fai_tensor=ones(N_f,1)*fai';
N_sample=500;
Input=zeros(N_x,N_y,N_sample);
Output=Input;
I=eye(2);
N_FFT=[Nfft_r,Nfft_a];
N_sig=[N_f,N_fai];
parfor c_sample=1:N_sample
    N_t = randi(N_t_max,1);
    coord_x=0.3*(rand(N_t,1)-0.5);            % coordinates generated randomly 
    coord_y=0.3*(rand(N_t,1)-0.5);
    amp_t=randn(N_t,1)+1j*randn(N_t,1);       % 
    Sig=zeros(N_f,N_fai);
    for c_t=1:N_t
        R_T=coord_x(c_t).*cosd(fai_tensor)+coord_y(c_t).*sind(fai_tensor);
        Sig=Sig+amp_t(c_t)*exp(-2j.*k_tensor.*R_T);
        Output(:,:,c_sample)=Output(:,:,c_sample)+abs(amp_t(c_t))*exp(-(X-coord_x(c_t)).^2/0.004^2-(Y-coord_y(c_t)).^2/0.004^2);
    end
    N_pad_pre=ceil((Nfft_r-N_f)/2);
    N_pad_post=Nfft_r-N_f-N_pad_pre;
    Sig=padarray(Sig,[N_pad_pre 0],0,'pre');
    Sig=padarray(Sig,[N_pad_post 0],0,'post');
    Sig=ifft(ifftshift(Sig,1),[],1);
    Sig=[Sig(end-N_x+ceil(N_x/2)+1:end,:);Sig(1:ceil(N_x/2),:)];
    N_pad_pre=ceil((Nfft_a-N_fai)/2);
    N_pad_post=Nfft_a-N_fai-N_pad_pre;
    Sig=padarray(Sig,[0 N_pad_pre],0,'pre');
    Sig=padarray(Sig,[0 N_pad_post],0,'post');
    Sig=ifft(ifftshift(Sig,2),[],2);
    Sig=[Sig(:,end-N_y+ceil(N_y/2)+1:end) Sig(:,1:ceil(N_y/2))];
    Input(:,:,c_sample)=Sig;
end
Input=single(Input);
Output=single(Output);