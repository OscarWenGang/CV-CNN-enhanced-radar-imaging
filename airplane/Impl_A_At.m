function out = Impl_A_At(s,N_FFT,N_sig,N_Ima,mode)
% 程序说明：用于隐式地实现线性算子A和A'，用于对801转台数据进行稀疏恢复成像
% Input:
%   s —— 输入信号 if mode == 1, out = A * s, elseif mode == 2, out = A' * s
%   N_FFT —— 包含两个元素，分别为第一和第二维度的FFT点数
%   N_sig —— 包含两个元素，分别为第一(N_f)和第二(N_fai)维度回波信号的点数
%   N_Ima —— 包含两个元素，分别为第一(x,与f对应)和第二(y,与fai对应)维度的图像点数
%   mode —— 1：回波生成模式， 2：成像模式
% Output：
%   out —— 输出信号 if mode == 1, out = A * s, elseif mode == 2, out = A' * s

% Author：高敬坤  Date：2017.12.11
I=eye(2);
if mode == 1                 % 回波生成模式
    s=reshape(s,[N_Ima(1) N_Ima(2)]);   % 将s变为矩阵形式，第一维：N_x，第二维：N_y
    c_t=1;
    N_pad_pre=ceil((N_FFT(c_t)-N_Ima(c_t))/2);
    N_pad_post=N_FFT(c_t)-N_Ima(c_t)-N_pad_pre;
    s=padarray(s,N_pad_pre*I(c_t,:),0,'pre');
    s=padarray(s,N_pad_post*I(c_t,:),0,'post');
    s=fft(ifftshift(s,c_t),[],c_t)/sqrt(N_FFT(c_t));
    s=[s(end-N_sig(1)+ceil(N_sig(1)/2)+1:end,:);s(1:ceil(N_sig(1)/2),:)];
    c_t=2;
    N_pad_pre=ceil((N_FFT(c_t)-N_Ima(c_t))/2);
    N_pad_post=N_FFT(c_t)-N_Ima(c_t)-N_pad_pre;
    s=padarray(s,N_pad_pre*I(c_t,:),0,'pre');
    s=padarray(s,N_pad_post*I(c_t,:),0,'post');
    s=fft(ifftshift(s,c_t),[],c_t)/sqrt(N_FFT(c_t));
    s=[s(:,end-N_sig(2)+ceil(N_sig(2)/2)+1:end) s(:,1:ceil(N_sig(2)/2))];
elseif mode == 2             % 成像模式
    s=reshape(s,[N_sig(1) N_sig(2)]);   % 将s变为矩阵形式，第一维：N_f，第二维：N_fai
    c_t=1;
    N_pad_pre=ceil((N_FFT(c_t)-N_sig(c_t))/2);
    N_pad_post=N_FFT(c_t)-N_sig(c_t)-N_pad_pre;
    s=padarray(s,N_pad_pre*I(c_t,:),0,'pre');
    s=padarray(s,N_pad_post*I(c_t,:),0,'post');
    s=ifft(ifftshift(s,c_t),[],c_t)*sqrt(N_FFT(c_t));
    s=[s(end-N_Ima(1)+ceil(N_Ima(1)/2)+1:end,:);s(1:ceil(N_Ima(1)/2),:)];
    c_t=2;
    N_pad_pre=ceil((N_FFT(c_t)-N_sig(c_t))/2);
    N_pad_post=N_FFT(c_t)-N_sig(c_t)-N_pad_pre;
    s=padarray(s,N_pad_pre*I(c_t,:),0,'pre');
    s=padarray(s,N_pad_post*I(c_t,:),0,'post');
    s=ifft(ifftshift(s,c_t),[],c_t)*sqrt(N_FFT(c_t));
    s=[s(:,end-N_Ima(2)+ceil(N_Ima(2)/2)+1:end) s(:,1:ceil(N_Ima(2)/2))];
end
out=s(:);
end

