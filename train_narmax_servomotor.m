% Clear Data
clc;
clear variables;
clear;
warning off MATLAB:subscripting:noSubscriptsSpecified

% initiate network architecture and prediction horizon
delay = 8;
neurons = 10;
train_k = 100;
train_o = 40;

% import training data
[u_pre, y_pre] = import_data_servomotor();

% train narmax
fprintf('\nTRAINING NARMAX MODEL:\n');

narmax = NARMAXmodel(delay, neurons);
narmax.trainAlg = 'trainlm';
narmax.earlyStoppage = false;
narmax.iterPerRun = 4000;
narmax.iterAfterValley = 100;
narmax.iterAfterSeq = 50;
narmax.maxStep = 10;
narmax.fill = 2;
narmax.initialTraining = 10;
narmax.zero_input_delay = false;

narmax = narmax.train(u_pre, y_pre, train_k);
dir = sprintf('train/servomotor/narmax_%sdelay_%sneurons_%ssteps', num2str(delay), num2str(neurons), num2str(train_k));
mkdir(dir);
save(sprintf('%s/narmax_model', dir) , 'narmax');
