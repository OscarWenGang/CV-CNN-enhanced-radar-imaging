function [ netGrad, eE ] = JK_backProp_nn_r( netNet, netTopo, netFture, netA, outPut )
% Input£º
%   netNet
%   netTopo 
%   netFture 
%   netA
%   outPut
% Output£º
%   netGrad
%   eE
nL=numel(netTopo); 
eE=mean(sum(sum((netFture{nL+1}-outPut).^2,1),2),4);
eD=netFture{nL+1}-outPut;
delR=cell(nL,1);
netGrad=cell(nL,1);
batchS=size(outPut,4);
switch netTopo{nL}.nonLineType
    case 'none'  
        delR{nL}=eD;
    otherwise
end
for cl=nL-1:-1:1
    if netTopo{cl+1}.dilate<2
        tempR=netNet{cl+1}.k;
    else
        kS=size(netNet{cl+1}.k);
        kS(1)=kS(1)+(kS(1)-1)*(netTopo{cl}.dilate-1);
        kS(2)=kS(2)+(kS(2)-1)*(netTopo{cl}.dilate-1);
        tempR=zeros(kS,'single');
        tempR(1:netTopo{cl}.dilate:end,1:netTopo{cl}.dilate:end,:,:)=real(netNet{cl+1}.k);
    end
    delR{cl}=vl_nnconvt(delR{cl+1},tempR,[]);
    switch netTopo{cl}.nonLineType
        case 'rReLU'
            delR{cl}(netA{cl}<0)=0;
        case 'rReLUleaky'
            delR{cl}(netA{cl}<0)=0.5e-1*delR{cl}(netA{cl}<0);
        case 'rSigm'
            delR{cl}=delR{cl}.*netFture{cl+1}.*(1-netFture{cl+1});
        case 'rTanh'
            delR{cl}=delR{cl}.*(1-netFture{cl+1}.^2);
        otherwise
    end
end
clear eD outPut netA;
for cl=nL:-1:1
    netFture{cl}=ipermute(netFture{cl},[1 2 4 3]);
    delR{cl}=ipermute(delR{cl},[1 2 4 3]);
    for co=1:netTopo{cl}.outFtureN
        for ci=1:netTopo{cl}.inFtureN
            netGrad{cl}.dk(:,:,ci,co)=vl_nnconv(netFture{cl}(:,:,:,ci),delR{cl}(:,:,:,co),[],'stride',netTopo{cl}.dilate)/batchS;
        end
        netGrad{cl}.db(co)=sum(sum(sum(delR{cl}(:,:,:,co),1),2),3)/batchS;
    end
    netFture{cl}=[];
    delR{cl}=[];
end
if gather(all(isnan(netGrad{nL}.dk(:,1,1,1))))
    warning('NAN appers');
end
if eE>1e3
    grad_scale=10^(-ceil(log10(eE/1e3)));
    for cl=nL:-1:1
        for co=1:netTopo{cl}.outFtureN
            for ci=1:netTopo{cl}.inFtureN
                netGrad{cl}.dk(:,:,ci,co)=grad_scale*netGrad{cl}.dk(:,:,ci,co);
            end
            netGrad{cl}.db(co)=grad_scale*netGrad{cl}.db(co);
        end
    end
end
end