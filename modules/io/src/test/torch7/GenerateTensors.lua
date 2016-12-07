----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'
require 'boof'

local operation_name = "tensor_storage"


for k,data_type in pairs(boof.all_types) do
    torch.setdefaulttensortype(boof.boof_to_tensor_name(data_type))

    local output_dir = boof.create_output(operation_name,data_type,1)

    local tensor = torch.zeros(3,20)
    local s = tensor:storage()
    for i=1,s:size() do -- fill up the Storage
        s[i] = i
    end

    torch.save(paths.concat(output_dir,'tensor'), tensor)
    torch.save(paths.concat(output_dir,'tensor_ascii'), tensor, 'ascii')
end