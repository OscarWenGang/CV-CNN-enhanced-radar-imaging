% Train the CCNN for radar imaging
% Author£ºGao Jingkun  Date£º2017.10.08
clear;
%% network topology and hyper parameters for training
netTopo={
	struct('layerType','c','inFtureN',1 ,'outFtureN',6 ,'kernelS',25,'dilate',1,'nonLineType','cReLU') %   cReLU   cSigm   cTanh   cReLUleaky
	struct('layerType','c','inFtureN',6 ,'outFtureN',12,'kernelS',15,'dilate',1,'nonLineType','cReLU')
    struct('layerType','c','inFtureN',12,'outFtureN',12,'kernelS',5 ,'dilate',1,'nonLineType','cReLU')
	struct('layerType','c','inFtureN',12,'outFtureN',1 ,'kernelS',3 ,'dilate',1,'nonLineType','Abs' )
	};
	
% 'layerType'   -- type for current layer
% 'inFtureN'    -- number of input feature maps for current layer
% 'outFtureN'   -- number of output feature maps for current layer
% 'kernelS'     -- size of the kernel(filter)
% 'nonLineType' -- type of the activation function

netTrainParas=struct('batchS',50,...                              % batchsize 
					'lM','SGDM_WD',...                            % learning method, 'SGD', 'SGDM'- SGD with momentum, 'SGDM_WD' - SGDM with weight decay
					'lR',3e-5*ones(numel(netTopo),1),...          % learning rate
					'errorFunc','RMS',...                         % objective
                    'epochN',5);                                  % epochs
% netTrainParas.lR(1)=1e-4;
netTrainParas.lR(end)=1e-5;

[ netNet ] = JK_setUp_nn(netTopo);                                % initialization
%% training
folders=['TraData';];
N_block=100;                                                      % number of training blocks in each folder
N_sample = 500;                                                   % number of training examples in each block
N_batch = N_sample/netTrainParas.batchS;                          % number of batches in each training block
samplesize = 180;
padsize = 0;
outsize = samplesize+2*padsize;
for cl=1:numel(netTopo)
    outsize=outsize-netTopo{cl}.kernelS-(netTopo{cl}.dilate-1)*(netTopo{cl}.kernelS-1)+1;  % kernerS is default to be odd
end
shrinksize = (samplesize - outsize)/2;
errors=zeros(N_batch,N_block,numel(folders),netTrainParas.epochN); % record the value of the objective function
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
            sampleOrder=randperm(N_sample);  % shuffle 
            for cbatch=1:N_batch
                inPut=permute(gpuArray(complex(inPutsamples(:,:,sampleOrder((cbatch-1)*netTrainParas.batchS+1:cbatch*netTrainParas.batchS)))),[1 2 4 3]);
                outPut=permute(gpuArray(outPutsamples(:,:,sampleOrder((cbatch-1)*netTrainParas.batchS+1:cbatch*netTrainParas.batchS))),[1 2 4 3]);
                % feedforward
                [ netFture, netA ] = JK_feedForward_nn( netNet, netTopo, inPut );
                % backpropagation
                [ netGrad, eE ] = JK_backProp_nn( netNet, netTopo, netFture, netA, outPut );
                errors(cbatch,cblock,cfolder,cepoch)=gather(eE);
                % updateNet
                [ netNet, netGradM ] = JK_updateNet_nn( netNet, netTopo, netGrad, netGradM, netTrainParas.lR, netTrainParas.lM );
            end
        end
    end
end
rmpath('..\dependence\mex');
%% save the trained network
save CCNN_net blockOrder cbatch cblock cepoch cfolder errors N_batch N_block N_sample netA netFture netGrad netGradM netNet netTopo netTrainParas
