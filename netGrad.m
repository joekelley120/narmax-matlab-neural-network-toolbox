function [gWB,perf,N] = netGrad(net,x,xi,ai,t,ew,mask)
%NETGRAD Calculation gradient for a neural network
%
% [gWB,perf,N] = netGrad(net,x,xi,ai,t,ew,mask), takes:
%   net - neural network
%   x - inputs (matrix or cell array of matrices)
%   xi - input states (cell array of matrices, or empty matrix)
%   ai - layer states (cell array of matrices, or empty matrix)
%   t - targets (matrix or cell array of matrices)
%   ew - Error weights (matrix or cell array of matrices or empty matrix)
%   mask - Target mask (matrix or cell array of matrices of 1/NaN values, or empty matrix)
% and returns:
%   gWB - derivative of performance with respect to weights and biases
%   perf - performance
%   N - number of performance values used for perf and gWB
%
% Wherever an input argument is allowed to have an empty matrix value, that
% input argument will be set to default values:
%   xi => zeros
%   ai => zeros
%   ew => {1}
%   mask => ones same size as t
%
% Static Example:
%
%  [x,t] = house_dataset;
%  net = feedforwardnet(10);
%  net = configure(net,x,t);
%  [gWB,perf,N] = netGrad(net,x,[],[],t)
%
% Dynamic Example:
%
%  [x,t] = maglev_dataset;
%  net = narxnet(1:2,1:2,10);
%  [X,Xi,Ai,T] = preparets(net,x,{},t);
%  net = configure(net,X,T);
%  [gWB,perf,N] = netGrad(net,X,Xi,Ai,T)

is = nn.input_sizes(net);
ls = nn.layer_sizes(net);
os = nn.output_sizes(net);
nid = net.numInputDelays;
nld = net.numLayerDelays;
if iscell(x), Q = size(x{1},2); else Q = size(x,2); end
if iscell(x), TS = size(x,2); else TS = 1; end

if (nargin < 3) || isempty(xi), xi = nndata(is,Q,nid,0); end
if (nargin < 4) || isempty(ai), ai = nndata(ls,Q,nld,0); end
if (nargin < 6) || isempty(ew), ew = {1}; end
if (nargin < 7) || isempty(mask), mask = nndata(os,Q,TS,1); end

if ~iscell(x), x = {x}; end
if ~iscell(t), t = {t}; end
if ~iscell(ew), ew = {ew}; end
if ~iscell(mask), mask = {mask}; end

data.X = x;
data.Xi = xi;
data.Pc = {};
data.Pd = {};
data.Ai = ai;
data.T = t;
data.EW = ew;
data.Q = size(x{1},2);
data.TS = size(x,2);
data.train.enabled = true;
data.train.mask = mask;
data.val.enabled = false;
data.val.mask = nndata(os,Q,TS,0);
data.test.mask = false;
data.test.mask = nndata(os,Q,TS,0);

% calcMode = nnMATLAB;
calcMode = nnMex;
[calcLib,calcNet] = nncalc.setup(calcMode,net,data);
[gWB,perf,N] = calcLib.grad(calcNet);
