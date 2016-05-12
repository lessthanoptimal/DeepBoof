----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

operation_name = "spatial_dropout"

local function generate( variant , data_type, v1)
    local output_dir = boof.create_output(operation_name,data_type,variant)

    local input = torch.randn(1,3,16,14)
    local operation = nn.SpatialDropout(0.4,v1)
    operation:evaluate()
    local output = operation:forward(input)

    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), operation)
    torch.save(paths.concat(output_dir,'output'), output)
end

for k,data_type in pairs(boof.float_types) do
    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    generate(1,data_type,false)
    generate(2,data_type,true)
end