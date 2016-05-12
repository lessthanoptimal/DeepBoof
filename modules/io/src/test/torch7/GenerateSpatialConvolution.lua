----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

operation_name = "spatial_convolution"

W = 16
H = 17

nIn = 3   -- channels
nOut = 6
kW = 3    -- kernel
kH = 4
dW = 1    -- step
dH = 1
padW = 2  -- padding
padH = 1

numbatch = 2

local function generate( variant , data_type)
    local output_dir = boof.create_output(operation_name,data_type,variant)

    local input = torch.randn(numbatch,nIn,W,H)

    local operation = nn.SpatialConvolution(nIn, nOut,kW, kH, dW, dH, padW, padH)

    operation.weight = torch.randn(nOut,nIn,kH,kW)
    operation.bias = torch.randn(nOut)

    operation:evaluate()
    local output = operation:forward(input)

    -- Strip away useless parameters to cut down on file size
    operation.output = nil
    operation.gradBias = nil
    operation.gradInput = nil
    operation.gradWeight = nil

    boof.save(output_dir,input,operation,output)

end

for k,data_type in pairs(boof.float_types) do
    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    dW = 1
    dH = 1
    generate(1,data_type)

    dW = 2
    dH = 2
    generate(2,data_type)
end