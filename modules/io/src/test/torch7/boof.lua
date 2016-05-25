
----------------------------------------------------------------------
-- Various utility functions for creating unit tests in Torch for DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------

boof = {}


boof.output_base_directory = "torch_layers"

boof.float_types = {}
boof.float_types[1] = "F32"
boof.float_types[2] = "F64"

function boof.boof_to_tensor_name( boof_type )
    if boof_type == "F32" then
        return "torch.FloatTensor"
    elseif boof_type == "F64" then
        return "torch.DoubleTensor"
    elseif boof_type == "cuda" then
        return "torch.CudaTensor"
    else
        print("Unknown/Unsupported type "..boof_type)
    end
end

function boof.create_output(operation_name, data_type, variant)
    local output_dir = paths.concat(boof.output_base_directory,operation_name,string.format('%s/%03d',data_type,variant))
    os.execute("mkdir -p "..output_dir)
    print(output_dir)

    return output_dir
end

function boof.save( output_dir, input, operation, output )
    torch.save(paths.concat(output_dir,'input'), input)
    torch.save(paths.concat(output_dir,'operation'), operation)
    torch.save(paths.concat(output_dir,'output'), output)

    torch.save(paths.concat(output_dir,'input_ascii'), input, 'ascii')
    torch.save(paths.concat(output_dir,'operation_ascii'), operation, 'ascii')
    torch.save(paths.concat(output_dir,'output_ascii'), output, 'ascii')
end
