----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

operation_name = "spatial_max_pooling"

numbatch = 2
C = 3
W = 16
H = 17

local function generate( variant , data_type)
    local output_dir = boof.create_output(operation_name,data_type,variant)

    local input = torch.randn(numbatch,C,W,H)

    local operation = nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)

    operation:evaluate()
    local output = operation:forward(input)

    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), operation)
    torch.save(paths.concat(output_dir,'output'), output)

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