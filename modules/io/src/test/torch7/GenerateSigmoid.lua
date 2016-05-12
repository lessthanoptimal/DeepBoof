----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

local operation_name = "sigmoid"
local variant = 1

for k,data_type in pairs(boof.float_types) do
    local output_dir = boof.create_output(operation_name,data_type,variant)

    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    local input = torch.randn(1,5,4,3)
    local operation = nn.Sigmoid()

    operation:evaluate()
    local output = operation:forward(input)

    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), operation)
    torch.save(paths.concat(output_dir,'output'), output)
end