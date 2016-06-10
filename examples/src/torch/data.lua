----------------------------------------------------------------------
-- This script demonstrates how to load the Face Detector 
-- training data, and pre-process it to facilitate learning.
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet, Eugenio Culurciello
-- Mon Oct 14 14:58:50 EDT 2013
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator

local opt = opt or {
   visualize = false,
   size = 'small',
   patches='all'
}

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> loading dataset')

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

train_dir = '../..'

print("training directory "..train_dir)
if not file_exists(train_dir .. '/train_normalized_cifar10.t7') then
    print("Please generate normalized input data using java examples")
    os.exit(1)
end

local trainset = torch.load(train_dir..'/train_normalized_cifar10.t7')
local testset = torch.load(train_dir..'/test_normalized_cifar10.t7')
local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print("Train set size: " .. trainset.data:size(1) .. " test set size " .. testset.data:size(1))
print("      set size: " .. trainset.label:storage():size() .. " test set size " .. testset.label:storage():size())

setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);

function trainset:size()
    return self.data:size(1)
end

function testset:size()
    return self.data:size(1)
end


function reduceData( dataHolder , fraction)
  local dataShuffle = torch.randperm(dataHolder:size(1))
  
  local smallSize = torch.floor(dataHolder:size(1)*fraction)
  local tmp = {}

  local labelStorage = dataHolder.label:storage()

  tmp.data = torch.Tensor(smallSize, dataHolder.data:size(2), dataHolder.data:size(3), dataHolder.data:size(4))
  tmp.label = torch.Tensor(smallSize)

  for i=1,smallSize do
    tmp.data[i] = dataHolder.data[dataShuffle[i]]
    tmp.label[i] = labelStorage[dataShuffle[i]]
  end

  function tmp:size()
    return self.data:size(1)
  end
  
  return tmp
end

-- Should be 1 indexed not 0
trainset.label:add(1)
testset.label:add(1)

-- Adjust size of training set based on request
if opt.size == 'small' then
  trainset = reduceData(trainset,0.1)
  testset = reduceData(testset,0.15)

  print("Reduced size of sets to ",trainset:size()," ",testset:size())
end

-- other parts of the code uses labels not label
trainset.labels = trainset.label
testset.labels = testset.label

-- Exports
return {
   trainData = trainset,
   testData = testset,
   classes = classes
}
