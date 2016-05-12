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

train_dir = 'input_data'

if not file_exists(train_dir .. '/cifar10-train.t7') then
   os.execute('mkdir -p ' .. train_dir)
   os.execute('cd ' .. train_dir)
   os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip -P ' .. train_dir)
   os.execute('unzip '.. train_dir ..'/cifar10torchsmall.zip -d ' .. train_dir)
end

local trainset = torch.load(train_dir..'/cifar10-train.t7')
local testset = torch.load(train_dir..'/cifar10-test.t7')
local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print("Train set size: " .. trainset.data:size(1) .. " test set size " .. testset.data:size(1))

setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);
testset.data = testset.data:double()
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size()
    return self.data:size(1)
end

function testset:size()
    return self.data:size(1)
end

print(sys.COLORS.red ..  '==> normalizing dataset')

local mean = {} -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

if opt.visualize == true then
  print("WTF it's visualizing? ",opt.visualize)
  image.display{image=testset.data[100], legend='test 100'}
end


function reduceData( dataHolder , fraction)
  local dataShuffle = torch.randperm(dataHolder:size(1))
  
  local smallSize = torch.floor(dataHolder:size(1)*fraction)
  local tmp = {}
  tmp.data = torch.Tensor(smallSize, dataHolder.data:size(2), dataHolder.data:size(3), dataHolder.data:size(4))
  tmp.label = torch.Tensor(smallSize)
  
  for i=1,smallSize do
    tmp.data[i] = dataHolder.data[dataShuffle[i]]
    tmp.label[i] = dataHolder.label[dataShuffle[i]]
  end

  function tmp:size()
    return self.data:size(1)
  end
  
  return tmp
end

-- Adjust size of training set based on request
if opt.size == 'small' then
  trainset = reduceData(trainset,0.25)
  testset = reduceData(testset,0.25)

  print("Reduced size of sets to ",trainset:size()," ",testset:size())
end

-- other parts of the code uses labels not label
trainset.labels = trainset.label
testset.labels = testset.label

-- Exports
return {
   trainData = trainset,
   testData = testset,
   mean = mean,
   std = std,
   classes = classes
}
