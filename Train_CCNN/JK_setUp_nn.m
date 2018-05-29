function [ netNet ] = JK_setUp_nn(netTopo)
% function: initialize the network
% Input£º
%     netTopo -- topology of the network
% Output£º
%     netNet
nL=numel(netTopo);
netNet=cell(nL,1);
for cl=1:nL
    switch netTopo{cl}.layerType
        case 'c'
            fan = netTopo{cl}.inFtureN * netTopo{cl}.outFtureN * netTopo{cl}.kernelS ^ 2;
            netNet{cl}.k=gpuArray(zeros(netTopo{cl}.kernelS,netTopo{cl}.kernelS,netTopo{cl}.inFtureN,netTopo{cl}.outFtureN,'single'));
            for co = 1: netTopo{cl}.outFtureN
                for ci = 1: netTopo{cl}.inFtureN
                    netNet{cl}.k(:,:,ci,co)=gpuArray(complex(randn(netTopo{cl}.kernelS,'single')/sqrt(fan*2),...
                     randn(netTopo{cl}.kernelS,'single')/sqrt(fan*2)));
                end
                netNet{cl}.b(co) = gpuArray(zeros(1,'single'));
            end
        otherwise
    end
end
end