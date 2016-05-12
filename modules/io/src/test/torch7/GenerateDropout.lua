----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

operation_name = "dropout"

local function generate( variant , data_type, v1)
    local output_dir = boof.create_output(operation_name,data_type,variant)

    local input = torch.randn(1,20)
    local operation = nn.Dropout(0.4,v1)
    operation:evaluate()
    local output = operation:forward(input)

    boof.save(output_dir,input,operation,output)
end

for k,data_type in pairs(boof.float_types) do
    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    generate(1,data_type,false)
    generate(2,data_type,true)
end