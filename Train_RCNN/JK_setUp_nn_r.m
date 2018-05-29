function [ netNet ] = JK_setUp_nn_r(netTopo)
% Input£º
%     netTopo 
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
                    netNet{cl}.k(:,:,ci,co)=gpuArray(randn(netTopo{cl}.kernelS,'single')/sqrt(fan));
                end
                netNet{cl}.b(co) = gpuArray(zeros(1,'single'));
            end
        otherwise
    end
end
end