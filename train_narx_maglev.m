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
[u_pre, y_pre] = import_data_maglev();

% train narx
fprintf('\nTRAINING NARX MODEL:\n');

narx = NARXmodel(delay, neurons);
narx.trainAlg = 'trainlm';
narx.earlyStoppage = true;
narx.iterPerRun = 4000;
narx.iterAfterValley = 100;
narx.iterAfterSeq = 50;
narx.maxStep = 10;
narx.initialTraining = 10;
narx.zero_input_delay = false;

narx = narx.train(u_pre, y_pre, train_k);
dir = sprintf('train/maglev/narx_%sdelay_%sneurons_%ssteps', num2str(delay), num2str(neurons), num2str(train_k));
mkdir(dir);
save(sprintf('%s/narx_model', dir), 'narx');
