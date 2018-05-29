function [ netFture, netA ] = JK_feedForward_nn( netNet, netTopo, inPut )
% Input£º
%   netNet
%   netTopo
%   inPut -- FtureS1 * FtureS2 * channelN * batchN
% Output£º
%   netFtures 
%   netA
nL=numel(netTopo);  % number of layers
netFture=cell(nL+1,1);
netA=cell(nL,1);
netFture{1}=inPut;
batchS=size(inPut,4);
for cl=1:nL
    switch netTopo{cl}.layerType
        case 'c'
            tempR=vl_nnconv(real(netFture{cl}),real(netNet{cl}.k),[],'dilate',netTopo{cl}.dilate)-vl_nnconv(imag(netFture{cl}),imag(netNet{cl}.k),[],'dilate',netTopo{cl}.dilate);
            tempI=vl_nnconv(real(netFture{cl}),imag(netNet{cl}.k),[],'dilate',netTopo{cl}.dilate)+vl_nnconv(imag(netFture{cl}),real(netNet{cl}.k),[],'dilate',netTopo{cl}.dilate);
            netA{cl}=tempR+1j*tempI;
            netFture{cl+1}=gpuArray(complex(zeros(size(netFture{cl},1)-size(netNet{cl}.k,1)+1,...
                size(netFture{cl},2)-size(netNet{cl}.k,2)+1,netTopo{cl}.outFtureN,batchS,'single')));
            for co=1:netTopo{cl}.outFtureN
                netA{cl}(:,:,co,:)=netA{cl}(:,:,co,:)+netNet{cl}.b(co);
            end
            switch netTopo{cl}.nonLineType
                case 'cReLU'
                    netFture{cl+1} =arrayfun(@cReLU,netA{cl});
                case 'cReLUleaky'
                    netFture{cl+1} =arrayfun(@cReLUleaky,netA{cl});
                case 'Abs'
                    netFture{cl+1} =complex(abs(netA{cl}));
                case 'cSigm'
                    netFture{cl+1} =arrayfun(@cSigm,netA{cl});
                case 'cTanh'
                    netFture{cl+1} =arrayfun(@cTanh,netA{cl});
                case 'none'
                    netFture{cl+1} =netA{cl};
                otherwise
            end
        otherwise
    end
end
if gather(all(isnan(netFture{nL+1}(:,1,1,1))))
    warning('NAN appers');
end
    function A=cReLU(A)
        re=real(A);
        im=imag(A);
        A=complex(max(re,0),max(im,0));
    end
    function A=cReLUleaky(A)
        re=real(A);
        im=imag(A);
        A=complex(max(re,0)+0.5e-1*min(re,0),max(im,0)+0.5e-1*min(im,0));
    end
    function A=cSigm(A)
        re=real(A);
        im=imag(A);
        A=complex(1./(1+exp(-re)),1./(1+exp(-im)));
    end
    function A=cTanh(A)
        re=real(A);
        im=imag(A);
        A=complex(tanh(re),tanh(im));
    end
end