require 'torch'
require 'nn'

if opt.type == 'cuda' then
   require 'cunn'
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> construct CNN')

-- This gets around 73.0% on test starting around epoc 140
--                  91.1% on training
--
--  the above was with some SGD configuration

-- Test = 68.8%  Train = 89.6% at epoc 75
-- th run.lua --search adam -r 0.00280 --adamBeta2 0.98489 --adamBeta1 0.70996 -l version005 -t 2 -s full -p cuda -b 125

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

ConvBatchReLUDrop(CNN,3,64,0.2)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 16
ConvBatchReLUDrop(CNN,64,128,0.4)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 8
ConvBatchReLUDrop(CNN,128,256,0.4)
ConvBatchReLUDrop(CNN,256,256,0.4)
ConvBatchReLUDrop(CNN,256,256,0.0)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 4
ConvBatchReLUDrop(CNN,256,512,0.4)
ConvBatchReLUDrop(CNN,512,512,0.0)
ConvBatchReLUDrop(CNN,512,512,0.0)
CNN:add(nn.SpatialMaxPooling(2,2,2,2)) -- 2
ConvBatchReLUDrop(CNN,512,512,0.4)
ConvBatchReLUDrop(CNN,512,512,0.4)
ConvBatchReLUDrop(CNN,512,512,0.0)
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
