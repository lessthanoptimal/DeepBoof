----------------------------------------------------------------------
-- Generates unit test data to test Torch to DeepBoof
--
-- Peter Abeles
----------------------------------------------------------------------



require 'torch'
require 'nn'

output_dir = "data"

os.execute("mkdir -p "..output_dir)

torch.setdefaulttensortype("torch.DoubleTensor")
torch.save(paths.concat(output_dir,'Tensor_F64'), torch.randn(2,3,10))
torch.setdefaulttensortype("torch.FloatTensor")
torch.save(paths.concat(output_dir,'Tensor_F32'), torch.randn(2,3,10))

