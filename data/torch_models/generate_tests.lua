-----------------------------------------------------------------------------------------------
-- Goes through each directory and creates input and output data for each network
-- so that compatibility with the current version of DeepBoof can be tested.
--
-- Usage: th generate_tests.lua
--
-- Peter Abeles
-----------------------------------------------------------------------------------------------


require 'lfs'

require 'torch'
require 'nn'
--require 'cunn'

local spatialInput = torch.randn(2,3,32,32):float()


for file in lfs.dir("./") do
    if lfs.attributes(file,"mode") == "directory" then
        local modelPath = paths.concat(file,"model.net")
        if lfs.attributes(modelPath,"mode") == "file" then
            print("Found network "..file)
            local network = torch.load(paths.concat(file,"model_float.net"))
            print("    ... running")
            network:evaluate()
            local output = network:forward(spatialInput)
            print("    ... saving")
            network.modules[1].weight:zero()
            torch.save(paths.concat(file,'model_float2.net'), network)

            torch.save(paths.concat(file,'test_input.t7'), spatialInput)
            torch.save(paths.concat(file,'test_output.t7'), output)
        end
    end
end

