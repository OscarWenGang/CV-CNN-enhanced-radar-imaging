function [ netGrad, eE ] = JK_backProp_nn( netNet, netTopo, netFture, netA, outPut )
% function: calculate the gradients using back-propagation
% Input£º
%   netNet    -- parameters of the network
%   netTopo   -- topology of the network
%   netFture  -- featuremaps
%   netA      -- input "a" of the activation function, where featuremap=h(a+b), where h() is the activation function
%   outPut    -- Groundtruth (labels)
% Output£º
%   netGrad   -- Gradients
%   eE
nL=numel(netTopo);                                       
eE=mean(sum(sum((netFture{nL+1}-outPut).^2,1),2),4); 
eD=netFture{nL+1}-outPut;
delR=cell(nL,1);
delI=cell(nL,1);
netGrad=cell(nL,1);
batchS=size(outPut,4);
switch netTopo{nL}.nonLineType
    case 'Abs'
        delO=netA{nL}./netFture{nL+1}.*eD;
        delO(isnan(delO))=0;
        delR{nL}=real(delO);
        delI{nL}=imag(delO);
    otherwise
end
for cl=nL-1:-1:1
    if netTopo{cl+1}.dilate<2
        tempR=real(netNet{cl+1}.k);
        tempI=imag(netNet{cl+1}.k);
    else
        kS=size(netNet{cl+1}.k);
        kS(1)=kS(1)+(kS(1)-1)*(netTopo{cl}.dilate-1);
        kS(2)=kS(2)+(kS(2)-1)*(netTopo{cl}.dilate-1);
        tempR=zeros(kS,'single');
        tempI=tempR;
        tempR(1:netTopo{cl}.dilate:end,1:netTopo{cl}.dilate:end,:,:)=real(netNet{cl+1}.k);
        tempI(1:netTopo{cl}.dilate:end,1:netTopo{cl}.dilate:end,:,:)=imag(netNet{cl+1}.k);
    end
    delR{cl}=vl_nnconvt(delR{cl+1},tempR,[])+vl_nnconvt(delI{cl+1},tempI,[]);
    delI{cl}=vl_nnconvt(delI{cl+1},tempR,[])-vl_nnconvt(delR{cl+1},tempI,[]);
    switch netTopo{cl}.nonLineType
        case 'cReLU'
            delR{cl}(real(netA{cl})<0)=0;
            delI{cl}(imag(netA{cl})<0)=0;
        case 'cReLUleaky'
            delR{cl}(real(netA{cl})<0)=0.5e-1*delR{cl}(real(netA{cl})<0);
            delI{cl}(imag(netA{cl})<0)=0.5e-1*delI{cl}(imag(netA{cl})<0);
        case 'cSigm'
            delR{cl}=delR{cl}.*real(netFture{cl+1}).*(1-real(netFture{cl+1}));
            delI{cl}=delI{cl}.*imag(netFture{cl+1}).*(1-imag(netFture{cl+1}));
        case 'cTanh'
            delR{cl}=delR{cl}.*(1-real(netFture{cl+1}).^2);
            delI{cl}=delI{cl}.*(1-imag(netFture{cl+1}).^2);
        otherwise
    end
end
clear eD delO outPut netA;
for cl=nL:-1:1
    netFture{cl}=ipermute(netFture{cl},[1 2 4 3]);
    delR{cl}=ipermute(delR{cl},[1 2 4 3]);
    delI{cl}=ipermute(delI{cl},[1 2 4 3]);
    for co=1:netTopo{cl}.outFtureN
        for ci=1:netTopo{cl}.inFtureN
            tempR=real(netFture{cl}(:,:,:,ci));
            tempI=imag(netFture{cl}(:,:,:,ci));
            tR=vl_nnconv(tempR,delR{cl}(:,:,:,co),[],'stride',netTopo{cl}.dilate)+vl_nnconv(tempI,delI{cl}(:,:,:,co),[],'stride',netTopo{cl}.dilate);
            tI=vl_nnconv(tempR,delI{cl}(:,:,:,co),[],'stride',netTopo{cl}.dilate)-vl_nnconv(tempI,delR{cl}(:,:,:,co),[],'stride',netTopo{cl}.dilate);
            netGrad{cl}.dk(:,:,ci,co)=(tR+1j*tI)/batchS;
        end
        netGrad{cl}.db(co)=sum(sum(sum(complex(delR{cl}(:,:,:,co),delI{cl}(:,:,:,co)),1),2),3)/batchS;
    end
    netFture{cl}=[];
    delR{cl}=[];
    delI{cl}=[];
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

