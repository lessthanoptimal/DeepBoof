-----------------------------------------------------------------------------------------------
-- Dumps the output for each layer in the network.  Only works with sequential networks
--
-- Usage: th dump_sequential_output.lua
--
-- Peter Abeles
-----------------------------------------------------------------------------------------------

require 'lfs'

require 'torch'
require 'nn'

local spatialInput = torch.randn(2,3,32,32):float()

local function processSequence( file, sequence , startIndex, inputTensor )
--    print("=============== ENTER processSequence() "..sequence.__typename)
    local output = inputTensor
    local num = startIndex
    for i=1,sequence:size() do
        local m = sequence.modules[i]

        print( "name = "..m.__typename )
        if m.__typename  == "nn.Sequential" then
            output,num = processSequence(file,m,num,output)
        else
            output = m:forward(output)
--            print("    ... saving "..num)
            torch.save(paths.concat(file,string.format('layer%03d.t7',num)), output)
        end
        num = num + 1
    end
    return output,num
end

for file in lfs.dir("./") do
    if lfs.attributes(file,"mode") == "directory" then
        local modelPath = paths.concat(file,"model.net")
        if lfs.attributes(modelPath,"mode") == "file" then
            print("Found network "..file)
            torch.save(paths.concat(file,'layers_input.t7'), spatialInput)
            local network = torch.load(paths.concat(file,"model_float.net")):float()
            network:evaluate()

            local output,num  = processSequence(file,network,1,spatialInput)
--            print(output)
--            output = network:forward(spatialInput)
--            print(output)
        end
    end
end