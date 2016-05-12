----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

operation_name = "sequential"

local function prune(lin)
    lin.output = nil
    lin.gradBias = nil
    lin.gradInput = nil
    lin.gradWeight = nil
end

------------------------------------------------------------------------------------------------------
----- Two linear operations in a row since the are easy to handle

local function generateLinearLinear( variant , data_type)
    local output_dir = boof.create_output(operation_name,data_type,variant)

    local input = torch.randn(2,20)

    local operation1 = nn.Linear(20,10,true)
    local operation2 = nn.Linear(10,6,true)

    operation1.weight = torch.randn(10,20)
    operation1.bias = torch.randn(10)
    operation2.weight = torch.randn(6,10)
    operation2.bias = torch.randn(6)

    local sequence = nn.Sequential()
    sequence:add(operation1)
    sequence:add(operation2)

    sequence:evaluate()
    local output = sequence:forward(input)

    -- Strip away useless parameters to cut down on file size
    prune(operation1)
    prune(operation2)
    sequence.output = nul

    -- No need to explicitly save the weight and bias because it is saved with the operation
    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), sequence)
    torch.save(paths.concat(output_dir,'output'), output)
end

------------------------------------------------------------------------------------------------------
----- See if nn.View is handled correctly

local function generateViewLinear( variant , data_type)
    local output_dir = boof.create_output(operation_name,data_type,variant)

    local input = torch.randn(2,3,15,13) -- Create a spatial input tensor

    local N = 3*15*13
    local operation1 = nn.View(N)
    local operation2 = nn.Linear(N,25,true)

    operation2.weight = torch.randn(25,N)
    operation2.bias = torch.randn(25)

    local sequence = nn.Sequential()
    sequence:add(operation1)
    sequence:add(operation2)

    sequence:evaluate()
    local output = sequence:forward(input)

    -- Strip away useless parameters to cut down on file size
    prune(operation1)
    prune(operation2)
    sequence.output = nul

    -- No need to explicitly save the weight and bias because it is saved with the operation
    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), sequence)
    torch.save(paths.concat(output_dir,'output'), output)
end

for k,data_type in pairs(boof.float_types) do
    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    generateLinearLinear(1,data_type)
    generateViewLinear(2,data_type)

end