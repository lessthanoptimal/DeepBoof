
local grid = {}

grid.minLearnRate = 1e-3
grid.maxLearnRate = 50.0

grid.minLearnDecay = 0.001 -- 0 is no change
grid.maxLearnDecay = 1

grid.minRegDecay = 1e-7
grid.maxRegDecay = 1e-2

grid.minMomentum = 0.0001 -- 0 is no change
grid.maxMomentum = 1.0

return {parameter_grid=grid,}