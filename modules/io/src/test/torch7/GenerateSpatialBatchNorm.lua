----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

local operation_name = "spatial_batch_normalization"

N = 2
C = 4
H = 8
W = 10

for k,data_type in pairs(boof.float_types) do
    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    local output_dir = boof.create_output(operation_name,data_type,1)

    local input = torch.randn(N,C,H,W)

    -- Create batch normalization with parameters that are not learnable
    local operation = nn.SpatialBatchNormalization(C, nil, nil, false)
    operation.running_mean = torch.randn(C)
    operation.running_var = torch.rand(C)

    operation:evaluate()
    local output = operation:forward(input)

    -- No need to explicitly save the weight and bias because it is saved with the operation
    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), operation)
    torch.save(paths.concat(output_dir,'output'), output)

    ------------------------------------------------------------------------
    -- The same but with gamma+beta (a.k.a. weight and bias)

    local output_dir = boof.create_output(operation_name,data_type,2)

    local operation = nn.SpatialBatchNormalization(C, nil, nil, true)
    operation.running_mean = torch.randn(C)
    operation.running_var = torch.rand(C)
    operation.weight = torch.randn(C)
    operation.bias = torch.rand(C)

    operation:evaluate()
    output = operation:forward(input)

    -- No need to explicitly save the weight and bias because it is saved with the operation
    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), operation)
    torch.save(paths.concat(output_dir,'output'), output)
end