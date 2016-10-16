----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

operation_name = "spatial_average_pooling"

numbatch = 2
C = 3
W = 16
H = 17

local function generate( variant , data_type)
    local output_dir = boof.create_output(operation_name,data_type,variant)

    local input = torch.randn(numbatch,C,W,H)

    local operation = nn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)

    operation:evaluate()
    local output = operation:forward(input)

    boof.save(output_dir,input,operation,output)

end

for k,data_type in pairs(boof.float_types) do
    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    kW = 3
    kH = 3
    dW = 1
    dH = 1
    padW = 1
    padH = 1
    generate(1,data_type)

    kW = 7
    kH = 5
    dW = 2
    dH = 1
    padW = 3
    padH = 2
    generate(2,data_type)
end