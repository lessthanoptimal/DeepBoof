----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

operation_name = "spatial_dropout"

for k,data_type in pairs(boof.float_types) do
    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    local output_dir = boof.create_output(operation_name,data_type,1)

    local input = torch.randn(1,3,16,14)
    local operation = nn.SpatialDropout(0.4)
    operation:evaluate()
    local output = operation:forward(input)

    boof.save(output_dir,input,operation,output)
end