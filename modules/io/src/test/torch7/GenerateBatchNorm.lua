----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

local operation_name = "batch_normalization"


for k,data_type in pairs(boof.float_types) do
    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    local output_dir = boof.create_output(operation_name,data_type,1)

    local input = torch.randn(3,20)

    -- Create batch normalization with parameters that are not learnable
    local operation = nn.BatchNormalization(20, nil, nil, false)
    operation.running_mean = torch.randn(20)
    operation.running_var = torch.rand(20)

    operation:evaluate()
    local output = operation:forward(input)

    -- No need to explicitly save the weight and bias because it is saved with the operation
    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), operation)
    torch.save(paths.concat(output_dir,'output'), output)

    ------------------------------------------------------------------------
    -- The same but with gamma+beta (a.k.a. weight and bias)

    local output_dir = boof.create_output(operation_name,data_type,2)

    local operation = nn.BatchNormalization(20, nil, nil, true)
    operation.running_mean = torch.randn(20)
    operation.running_var = torch.rand(20)
    operation.weight = torch.randn(20)
    operation.bias = torch.rand(20)

    operation:evaluate()
    output = operation:forward(input)

    -- No need to explicitly save the weight and bias because it is saved with the operation
    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), operation)
    torch.save(paths.concat(output_dir,'output'), output)
end