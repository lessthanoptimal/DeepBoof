require 'torch'
require 'nn'

if opt.type == 'cuda' then
   require 'cunn'
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> construct CNN')

-- This gets around 89.66% on test starting around epoc 140
--                  98.75% on training
--
-- th run.lua -r 0.29106 -sgdLearningRateDecay 0.0 --sgdMomentum 0.00152 -sgdWeightDecay 0.0 -s full --model version005 -p cuda -t 2

local function ConvBatchReLUDrop(CNN, input, output, dropFrac )
   CNN:add(nn.SpatialConvolution(input, output, 3,3, 1,1, 1,1))
   CNN:add(nn.SpatialBatchNormalization(output))
   CNN:add(nn.ReLU(true))
   if dropFrac > 0 then
      CNN:add(nn.SpatialDropout(dropFrac))
   end
end

-- http://torch.ch/blog/2015/07/30/cifar.html

local CNN = nn.Sequential()

ConvBatchReLUDrop(CNN,3,64,0.3)
ConvBatchReLUDrop(CNN,64,64,0.0)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 16

ConvBatchReLUDrop(CNN,64,128,0.4)
ConvBatchReLUDrop(CNN,128,128,0.0)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 8

ConvBatchReLUDrop(CNN,128,256,0.4)
ConvBatchReLUDrop(CNN,256,256,0.4)
ConvBatchReLUDrop(CNN,256,256,0.0)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 4

ConvBatchReLUDrop(CNN,256,512,0.4)
ConvBatchReLUDrop(CNN,512,512,0.4)
ConvBatchReLUDrop(CNN,512,512,0.0)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 2

ConvBatchReLUDrop(CNN,512,512,0.4)
ConvBatchReLUDrop(CNN,512,512,0.4)
ConvBatchReLUDrop(CNN,512,512,0.0)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 1

CNN:add(nn.View(512))

local classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU())
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 10))


local model = nn.Sequential()
model:add(CNN)
model:add(classifier)

-- Loss: NLL
local loss = nn.CrossEntropyCriterion()


-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  init'nn.SpatialConvolution'
end

MSRinit(CNN)


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

