----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'deepboof'

if opt.type == 'cuda' then
   require 'cunn'
end

----------------------------------------------------------------------
-- Model + Loss:

local t = require(opt.model)
local model = t.model
local loss = t.loss

local d = require 'data'
local classes = d.classes
local trainData = d.trainData

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining some tools')

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> flattening model parameters')

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> configuring optimizer')

local optimState = {}

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')
local x = torch.Tensor(opt.batchSize,trainData.data:size(2),
         trainData.data:size(3), trainData.data:size(4))
local yt = torch.Tensor(opt.batchSize)

if opt.type == 'cuda' then
   x = x:cuda()
   yt = yt:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')

local epoch

local function reset()
   epoch = 0

   if opt.search == 'sgd' then
      optimState = {
         learningRate = opt.learningRate,
         momentum = opt.sgdMomentum,
         weightDecay = opt.sgdWeightDecay,
         learningRateDecay = opt.sgdLearningRateDecay
      }
   elseif opt.search == 'adam' then
      optimState = {
         learningRate = opt.learningRate,
         beta1 = opt.adamBeta1,
         beta2 = opt.adamBeta2,
         epsilon = 1e-8
      }
   end
   -- reset weights
   model:reset()

   if  not trainLogger == nil then
      trainLogger.file:close()
   end
   trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
end

local function train(trainData)

   local file_param = io.open("results/training_parameters.txt", "w")

   if opt.search == 'sgd' then
      file_param:write('learningRate '..optimState.learningRate..'\n')
      file_param:write('momentum '..optimState.momentum..'\n')
      file_param:write('weightDecay '..optimState.weightDecay..'\n')
      file_param:write('learningRateDecay '..optimState.learningRateDecay..'\n')
   elseif opt.search == 'adam' then
      file_param:write('learningRate '..optimState.learningRate..'\n')
      file_param:write('beta1 '..optimState.beta1..'\n')
      file_param:write('beta2 '..optimState.beta2..'\n')
   end
   file_param:write('size '.. opt.size ..'\n')
   file_param:write('model '.. opt.model ..'\n')
   file_param:write('search '.. opt.search ..'\n')
   file_param:close()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData:size())

   -- Let it know that it's in training mode
   model:training()

   -- do one epoch
   print(sys.COLORS.green .. '==> doing epoch on training data:') 
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > trainData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         x[idx] = trainData.data[shuffle[i]]
         yt[idx] = trainData.labels[shuffle[i]]
         idx = idx + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local y = model:forward(x)
         local E = loss:forward(y,yt)

         -- estimate df/dW
         local dE_dy = loss:backward(y,yt)   
         model:backward(x,dE_dy)

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(y[i],yt[i])
         end

         -- print("E ",E," dE_dw ",dE_dw:sum()," w ",w:sum())
         -- return f and df/dX
         return E,dE_dw
      end

      -- optimize on current mini-batch
      optim[opt.search](eval_E, w, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   local file_confusion = io.open(paths.concat(opt.save , "confusion_human_train.txt"), "w")
   file_confusion:write(tostring(confusion))
   file_confusion:close()

   file_confusion = io.open(paths.concat(opt.save , "confusion_train.txt"), "w")
   file_confusion:write(deepboof.confusionToString(confusion))
   file_confusion:close()

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end


   -- next epoch
   local average_accuracy = confusion.totalValid
   confusion:zero()
   epoch = epoch + 1

   return average_accuracy,model:clone()
end

-- Export:
return {train=train,reset=reset}

