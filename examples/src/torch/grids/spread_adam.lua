
local grid = {}

grid.minLearnRate = 1e-6
grid.maxLearnRate = 0.5

grid.minBeta1 = 0.4
grid.maxBeta1 = 0.999

grid.minBeta2 = 0.8
grid.maxBeta2 = 0.99999

return {parameter_grid=grid,}