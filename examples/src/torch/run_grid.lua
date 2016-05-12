require 'pl'
require 'trepl'
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'lfs'
require 'posix.sys.stat'   -- luarocks install luaposix

----------------------------------------------------------------------

-- Adjust the path to allow dynamically loading of lua files

package.path = package.path .. ";".. lfs.currentdir().."/models/?.lua"
package.path = package.path .. ";".. lfs.currentdir().."/grids/?.lua"


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
   -g,--parameterGrid      (default spread_adam)  Which set of parameters to run
   -m,--model              (default version001)  which model to load
   -b,--batchSize          (default 128)         batch size
   -t,--threads            (default 1)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -s,--size               (default small)       dataset: small or full or extra
   -o,--save               (default results)     save directory
      --patches            (default all)         percentage of samples to use for testing'
      --visualize          (default false)       visualize dataset
      --search             (default sgd)         optimization algorithm (sgd,adam)
]]

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

print("parameter grid  " .. opt.parameterGrid)
print("model           " .. opt.model)
print("A random number " .. torch.random(10000))
print("data type       " .. opt.type)
print("search          " .. opt.search)

-- type:
if opt.type == 'cuda' then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
local data  = require 'data'
local g = require(opt.parameterGrid)
local parameter_grid = g.parameter_grid

print("Grid Parameters:")
print(parameter_grid)

function selectValue( minv , maxv)
    return math.random()*(maxv-minv)+minv
end

-- selects random numbers on a log scale
-- this is intended to increase the change of selecting very small to very large values
function selectValueLog( minv , maxv)
    if minv > maxv then
        print("minium is more than maximum! "..(minv).."  "..(maxv))
        os.exit(1)
    elseif minv > 0 then
        local range = math.log(maxv/minv)/math.log(10.0)
        local selected = math.random()*range
        return minv*math.pow(10,selected)
    else
        -- use linear since you can't use relative values for a minimum of 0
        return selectValue(minv,maxv)
    end
end

function isdir(fn)
    return not (posix.sys.stat.stat(fn) == nil)
end

local grid_trial = 1
local best_train = 0  -- fraction correct for training set
local best_test = 0   -- test set score

while true do
    print(sys.COLORS.red ..  '==> Configuring Parameters from Grid')


    if opt.search == 'sgd' then
        opt.learningRate      = selectValueLog(parameter_grid.minLearnRate  , parameter_grid.maxLearnRate)
        opt.learningRateDecay = selectValueLog(parameter_grid.minLearnDecay , parameter_grid.maxLearnDecay)
        opt.weightDecay       = selectValueLog(parameter_grid.minRegDecay   , parameter_grid.maxRegDecay)
        opt.momentum          = selectValueLog(parameter_grid.minMomentum   , parameter_grid.maxMomentum)
    elseif opt.search == 'adam' then
        opt.learningRate      = selectValueLog(parameter_grid.minLearnRate  , parameter_grid.maxLearnRate)
        opt.adamBeta1         = selectValueLog(parameter_grid.minBeta1      , parameter_grid.maxBeta1)
        opt.adamBeta2         = selectValueLog(parameter_grid.minBeta2      , parameter_grid.maxBeta2)
    end

    local ttrain = require 'train'
    local ttest  = require 'test'

    local train = ttrain.train
    local reset_train = ttrain.reset

    local test = ttest.test
    local reset_test = ttest.reset

----------------------------------------------------------------------
    print(sys.COLORS.red .. '==> training!')

    local local_best = 0        -- value of the best result locally
    local local_trial = 0       -- current trial number
    local tick_last_bested = 0  -- trial number when it last came close to the best value

    reset_train() -- let it know it's starting over again
    reset_test()

    while local_trial < 1000 and (local_trial- tick_last_bested) <= 10 do
        print('*********************  grid tick '..grid_trial.. ' | local tick '..(local_trial).." ticks since bested "..(local_trial- tick_last_bested))
        local results_train,model_train = train(data.trainData,best_test)
        local results_test  = test(data.testData)

        if results_test > local_best then
            tick_last_bested = local_trial
            local_best = results_test
        end

        -- Save the best results found so far
        if results_test > best_test+0.0001 then
            print('!!!!!!!! NEW BEST !!!!!!!!!!!!!')
            print("        score = "..results_test)
            best_test = results_test
            local path_best = paths.concat(opt.save, 'best')
            if not isdir(path_best) then -- only make a directory if it doesn't exist.  fewer errors this way
                os.execute("mkdir "..path_best)
            end

            -- save/log current net
            local model_file_name = paths.concat(path_best, 'model.net')
            local model1 = model_train:clone()
            torch.save(model_file_name, model1:clearState())

            os.execute('cp '.. opt.save ..'/*.log '..path_best)
            os.execute('cp '.. opt.save ..'/*.txt '..path_best)
        end

        -- If it's doing very poorly initially just give up
        if local_trial > 5 and results_test < 0.2 then
            break
        end

        local_trial = local_trial + 1
    end

    -- Save statistics from the local run
    print("Copying results")
    local grid_dir = paths.concat(opt.save,string.format('grid%06d',grid_trial))
    os.execute("mkdir "..grid_dir)
    os.execute('mv '.. opt.save ..'/*.log '..grid_dir)
    os.execute('mv '.. opt.save ..'/*.txt '..grid_dir)

    print("End cycle")
    grid_trial = grid_trial + 1
end