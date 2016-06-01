require 'torch'
require 'nn'

if opt.type == 'cuda' then
   require 'cunn'
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> construct CNN')

-- This gets around ????% on test starting around epoc ??
--                  ????% on training
--
-- th run.lua

local function ConvBatchDropReLU(CNN, input, output, dropFrac )
   CNN:add(nn.SpatialConvolution(input, output, 3,3, 1,1, 1,1))
   CNN:add(nn.SpatialBatchNormalization(output))
   CNN:add(nn.SpatialDropout(dropFrac))
   CNN:add(nn.ReLU())
end

local CNN = nn.Sequential()

ConvBatchDropReLU(CNN,3,64,0.2)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 16
ConvBatchDropReLU(CNN,64,128,0.4)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 8
ConvBatchDropReLU(CNN,128,256,0.4)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 4
ConvBatchDropReLU(CNN,256,512,0.4)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 2
ConvBatchDropReLU(CNN,512,512,0.4)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 1

CNN:add(nn.View(512))
CNN:add(nn.Dropout(0.5))

local classifier = nn.Sequential()
classifier:add(nn.Linear(512, 512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU())
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 10))
classifier:add(nn.LogSoftMax())


local model = nn.Sequential()
model:add(CNN)
model:add(classifier)

-- Loss: NLL
local loss = nn.ClassNLLCriterion()


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
}

