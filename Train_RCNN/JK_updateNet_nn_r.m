function [ netNet, netGradM ] = JK_updateNet_nn_r( netNet, netTopo, netGrad, netGradM, lR, lM )
% Input£º
%   netNet
%   netTopo
%   netGrad
%   netGradM
% Output£º
%   netNet
%   netGradM
nL=numel(netTopo);
for cl=1:nL
    for co=1:netTopo{cl}.outFtureN
        for ci=1:netTopo{cl}.inFtureN
            switch lM
                case 'SGD'
                    netNet{cl}.k(:,:,ci,co)=netNet{cl}.k(:,:,ci,co)-lR(cl)*netGrad{cl}.dk(:,:,ci,co);
                case 'SGDM'
                    netGradM{cl}.dk(:,:,ci,co)=0.9*netGradM{cl}.dk(:,:,ci,co)+lR(cl)*netGrad{cl}.dk(:,:,ci,co);
                    netNet{cl}.k(:,:,ci,co)=netNet{cl}.k(:,:,ci,co)-netGradM{cl}.dk(:,:,ci,co);
                case 'SGDM_WD'    % weight decay
                    netGradM{cl}.dk(:,:,ci,co)=0.9*netGradM{cl}.dk(:,:,ci,co)+lR(cl)*netGrad{cl}.dk(:,:,ci,co);
                    netNet{cl}.k(:,:,ci,co)=netNet{cl}.k(:,:,ci,co)-netGradM{cl}.dk(:,:,ci,co)-0.001*lR(cl)*netNet{cl}.k(:,:,ci,co);
                otherwise
            end
        end
        switch lM
            case 'SGD'
                netNet{cl}.b(co)=netNet{cl}.b(co)-lR(cl)*netGrad{cl}.db(co);
            case 'SGDM'
                netGradM{cl}.db(co)=0.9*netGradM{cl}.db(co)+lR(cl)*netGrad{cl}.db(co);
                netNet{cl}.b(co)=netNet{cl}.b(co)-netGradM{cl}.db(co);
            case 'SGDM_WD'
                netGradM{cl}.db(co)=0.9*netGradM{cl}.db(co)+lR(cl)*netGrad{cl}.db(co);
                netNet{cl}.b(co)=netNet{cl}.b(co)-netGradM{cl}.db(co)-0.001*lR(cl)*netNet{cl}.b(co);
            otherwise
        end
    end
end
end

