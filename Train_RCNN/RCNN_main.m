% Train a RCNN for radar imaging
% Author£ºGao Jingkun  Date£º2017.12.13
clear;
%% network topology and hyper parameters for training
netTopo={
	struct('layerType','c','inFtureN',2 ,'outFtureN',6 ,'kernelS',25,'dilate',1,'nonLineType','rReLU') %   rReLU   rSigm   rTanh   rReLUleaky
	struct('layerType','c','inFtureN',6 ,'outFtureN',24,'kernelS',15,'dilate',1,'nonLineType','rReLU')
    struct('layerType','c','inFtureN',24,'outFtureN',24,'kernelS',5 ,'dilate',1,'nonLineType','rReLU')
	struct('layerType','c','inFtureN',24,'outFtureN',1 ,'kernelS',3 ,'dilate',1,'nonLineType','none' )
	};
netTrainParas=struct('batchS',50,... 
					'lM','SGDM_WD',...
					'lR',3e-5*ones(numel(netTopo),1),...
					'errorFunc','RMS',...
                    'epochN',5);
netTrainParas.lR(end)=1e-5;
[ netNet ] = JK_setUp_nn_r(netTopo);
%%  training
folders=['TraData';];
N_block=100;
N_sample = 500;
N_batch = N_sample/netTrainParas.batchS; 
samplesize = 180;
padsize = 0;
outsize = samplesize+2*padsize;
for cl=1:numel(netTopo)
    outsize=outsize-netTopo{cl}.kernelS-(netTopo{cl}.dilate-1)*(netTopo{cl}.kernelS-1)+1;
end
shrinksize = (samplesize - outsize)/2;
errors=zeros(N_batch,N_block,numel(folders),netTrainParas.epochN); 
nL=numel(netTopo);
netGradM = cell(nL,1);
for cl=1:nL
    netGradM{cl}.dk=gpuArray(zeros(netTopo{cl}.kernelS,netTopo{cl}.kernelS,netTopo{cl}.inFtureN,netTopo{cl}.outFtureN,'single'));
    for co=1:netTopo{cl}.outFtureN
        netGradM{cl}.db(co)= gpuArray(zeros(1,'single'));
    end
end
addpath('..\dependence\mex');
for cepoch=1:netTrainParas.epochN
    for cfolder=1:length(folders)
        blockOrder=randperm(N_block);
        for cblock=1:N_block
            inPutsamples = importdata(['..\',folders(cfolder),'\Input_',num2str(blockOrder(cblock)) ,'.mat']);
            outPutsamples = importdata(['..\',folders(cfolder),'\Output_',num2str(blockOrder(cblock)) ,'.mat']);
            inPutsamples = single(padarray(inPutsamples*9,[padsize padsize],'replicate'));
            outPutsamples = single(outPutsamples(shrinksize+1:end-shrinksize,shrinksize+1:end-shrinksize,:));
            sampleOrder=randperm(N_sample);
            for cbatch=1:N_batch
                clear inPut;
				inPut(:,:,:,1)=gpuArray(real(inPutsamples(:,:,sampleOrder((cbatch-1)*netTrainParas.batchS+1:cbatch*netTrainParas.batchS))));
				inPut(:,:,:,2)=gpuArray(imag(inPutsamples(:,:,sampleOrder((cbatch-1)*netTrainParas.batchS+1:cbatch*netTrainParas.batchS))));
				inPut=permute(inPut,[1 2 4 3]);
                outPut=permute(gpuArray(outPutsamples(:,:,sampleOrder((cbatch-1)*netTrainParas.batchS+1:cbatch*netTrainParas.batchS))),[1 2 4 3]);
                % feedforward
                [ netFture, netA ] = JK_feedForward_nn_r( netNet, netTopo, inPut );
                % backpropagation
                [ netGrad, eE ] = JK_backProp_nn_r( netNet, netTopo, netFture, netA, outPut );
                errors(cbatch,cblock,cfolder,cepoch)=gather(eE);
                % updateNet
                [ netNet, netGradM ] = JK_updateNet_nn_r( netNet, netTopo, netGrad, netGradM, netTrainParas.lR, netTrainParas.lM );
            end
        end
    end
end
rmpath('..\dependence\mex');
%% save the trained network
save RCNN_net blockOrder cbatch cblock cepoch cfolder errors N_batch N_block N_sample netA netFture netGrad netGradM netNet netTopo netTrainParas
