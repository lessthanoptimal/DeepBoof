----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
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
print(sys.COLORS.red .. '==> defining some tools')

-- model:

local t = require(opt.model)
local model = t.model

local d = require 'data'
local classes = d.classes
local testData = d.testData

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes) -- faces: yes, no

-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Batch test:
local inputs = torch.Tensor(opt.batchSize,testData.data:size(2), 
         testData.data:size(3), testData.data:size(4)) -- get size from data
local targets = torch.Tensor(opt.batchSize)

if opt.type == 'cuda' then 
   inputs = inputs:cuda()
   targets = targets:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

local function reset()
   if  not testLogger == nil then
      testLogger.file:close()
   end
   testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
end

-- test function
function test(testData)
   -- local vars
   local time = sys.clock()

   -- put the network into evaluation mode
   model:evaluate()

   -- test over test data
   print(sys.COLORS.red .. '==> testing on test set:')
   for t = 1,testData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, testData:size())

      -- batch fits?
      if (t + opt.batchSize - 1) > testData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         inputs[idx] = testData.data[i]
         targets[idx] = testData.labels[i]
         idx = idx + 1
      end

      -- test sample
      local preds = model:forward(inputs)

      -- confusion
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   local file_confusion = io.open(paths.concat(opt.save , "confusion_human_test.txt"), "w")
   file_confusion:write(tostring(confusion))
   file_confusion:close()

   file_confusion = io.open(paths.concat(opt.save , "confusion_test.txt"), "w")
   file_confusion:write(deepboof.confusionToString(confusion))
   file_confusion:close()

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end
   local average_accuracy = confusion.totalValid
   confusion:zero()

   -- return accuracy
   return average_accuracy
end

-- Export:
return {test=test,reset=reset}

