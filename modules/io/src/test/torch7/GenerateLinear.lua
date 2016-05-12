----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

local operation_name = "linear"
local variant = 1

for k,data_type in pairs(boof.float_types) do
    local output_dir = boof.create_output(operation_name,data_type,variant)

    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    local input = torch.randn(1,20)

    local operation = nn.Linear(20,10,true)

    operation.weight = torch.randn(10,20)
    operation.bias = torch.randn(10)

    operation:evaluate()
    local output = operation:forward(input)

    -- Strip away useless parameters to cut down on file size
    operation.output = nil
    operation.gradBias = nil
    operation.gradInput = nil
    operation.gradWeight = nil

    -- No need to explicitly save the weight and bias because it is saved with the operation
    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), operation)
    torch.save(paths.concat(output_dir,'output'), output)
end