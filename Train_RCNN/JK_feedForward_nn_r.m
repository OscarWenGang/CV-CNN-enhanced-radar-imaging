function [ netFture, netA ] = JK_feedForward_nn_r( netNet, netTopo, inPut )
% Input£º
%   netNet
%   netTopo
%   inPut FtureS1 * FtureS2 * channelN * batchN
% Output£º
%   netFtures
%   netA 
nL=numel(netTopo); 
netFture=cell(nL+1,1);
netA=cell(nL,1);
netFture{1}=inPut;
batchS=size(inPut,4);
for cl=1:nL
    switch netTopo{cl}.layerType
        case 'c'
            netA{cl}=vl_nnconv(netFture{cl},netNet{cl}.k,[],'dilate',netTopo{cl}.dilate);
            netFture{cl+1}=gpuArray(zeros(size(netFture{cl},1)-size(netNet{cl}.k,1)+1,...
                size(netFture{cl},2)-size(netNet{cl}.k,2)+1,netTopo{cl}.outFtureN,batchS,'single'));
            for co=1:netTopo{cl}.outFtureN
                netA{cl}(:,:,co,:)=netA{cl}(:,:,co,:)+netNet{cl}.b(co);
            end
            switch netTopo{cl}.nonLineType
                case 'rReLU'
                    netFture{cl+1} =arrayfun(@rReLU,netA{cl});
                case 'rReLUleaky'
                    netFture{cl+1} =arrayfun(@rReLUleaky,netA{cl});
                case 'rSigm'
                    netFture{cl+1} =arrayfun(@rSigm,netA{cl});
                case 'rTanh'
                    netFture{cl+1} =arrayfun(@rTanh,netA{cl});
                case 'none'
                    netFture{cl+1}=netA{cl};
                otherwise
            end
        otherwise
    end
end
if gather(all(isnan(netFture{nL+1}(:,1,1,1))))
    warning('NAN appers');
end
    function A=rReLU(A)
        A=max(A,0);
    end
    function A=rReLUleaky(A)
        A=max(A,0)+0.5e-1*min(A,0);
    end
    function A=rSigm(A)
        A=1./(1+exp(-A));
    end
    function A=rTanh(A)
        A=tanh(A);
    end
end