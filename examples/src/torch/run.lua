----------------------------------------------------------------------------------------
-- Learns a network using a single set of parameters.  Typically this run against the
-- full set of training data after reasonable parameters have been found using the
-- grid search.
--
-- Authors:
--   Peter Abeles
--
----------------------------------------------------------------------

require 'pl'
require 'trepl'
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'posix.sys.stat'   -- luarocks install luaposix

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
   -r,--learningRate          (default 1e-3)        learning rate
      --sgdLearningRateDecay  (default 1e-7)        learning rate decay (in # samples)
      --sgdWeightDecay        (default 1e-5)        L2 penalty on the weights
      --sgdMomentum           (default 0.1)         momentum
      --adamBeta1             (default 0.9)         adam beta1 parameter
      --adamBeta2             (default 0.999)       adam beta2 parameter
   -l,--model                 (default version001)  which model to load
   -b,--batchSize             (default 128)         batch size
   -t,--threads               (default 1)           number of threads
   -p,--type                  (default float)       float or cuda
   -i,--devid                 (default 1)           device ID (if using CUDA)
   -s,--size                  (default small)       dataset: small or full or extra
   -o,--save                  (default results)     save directory
      --maxNoImprovement      (default 0)           How long it will run since it last improved.  0 to distable
      --patches               (default all)         percentage of samples to use for testing'
      --visualize             (default false)       visualize dataset
      --search                (default sgd)         optimization algorithm (sgd,adam)
]]

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

print("model           " .. opt.model)
print("A random number " .. torch.random(10000))
print("data type       " .. opt.type)
print("search          " .. opt.search)

-- type:
if opt.type == 'cuda' then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

package.path = package.path .. ";models/" .. opt.model .. '.lua'

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

local data  = require 'data'
local ttrain = require 'train'
local ttest  = require 'test'

local train = ttrain.train
local reset_train = ttrain.reset

local test = ttest.test
local reset_test = ttest.reset

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')

reset_train()
reset_test()

local best_test = 0
local epoc = 0
local last_best = 0

-- save all command line arguments to disk
torch.save(paths.concat(opt.save, 'arguments.t7'), opt, 'ascii')

while true do
   local results_train,model_train = train(data.trainData)
   local results_test  = test(data.testData)

   -- If the model is better then save it
   if best_test < results_test then
      best_test = results_test
      last_best = epoc
      print("Saving model.  score = "..best_test)
      local model_file_name = paths.concat(opt.save, 'model.net')
      local model1 = model_train:clone()
      torch.save(model_file_name, model1:clearState())
   end

   -- If performance hasn't improved in a long time stop
   if opt.maxNoImprovement > 0 and epoc - last_best >= opt.maxNoImprovement then
      break
   else
      print("Epocs since improvement "..(epoc - last_best).." max allowed "..opt.maxNoImprovement)
   end

   epoc = epoc + 1
end

print("Finished!")

