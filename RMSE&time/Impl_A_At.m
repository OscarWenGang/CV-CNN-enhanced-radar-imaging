function out = Impl_A_At(s,N_FFT,N_sig,N_Ima,mode)
% function£ºimplicitly implement A and A'
% Input:
%   s -- input signal  if mode == 1, out = A * s, elseif mode == 2, out = A' * s
%   N_FFT
%   N_si
%   N_Ima
%   mode -- 1£ºgenerate echo mode, 2£ºimaging mode
% Output£º
%   out -- output if mode == 1, out = A * s, elseif mode == 2, out = A' * s

% Author£ºGao Jingkun  Date£º2017.12.11
I=eye(2);
if mode == 1               
    s=reshape(s,[N_Ima(1) N_Ima(2)]); 
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
elseif mode == 2       
    s=reshape(s,[N_sig(1) N_sig(2)]);
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

