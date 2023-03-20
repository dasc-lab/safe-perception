# For storing point clouds in (more) efficient representations

# 2D occupancy grid
using StaticArrays

struct OccupancyGrid{T<:Real}
    start_x::T # world frame x-coord at (1, 1)
    start_y::T # world frame y-coord at (1, 1)
    cell_size::T # Size of each square grid cell
    grid::MMatrix  # Possible to specify Bool type but not dimensions?
end
OccupancyGrid{T}(num_x, num_y, start_x, start_y, cell_size) where {T<:Real} = OccupancyGrid{T}(start_x, start_y, cell_size, MMatrix{num_x, num_y, Bool}([false for i in 1:num_x*num_y]))

function add_points!(og, points)
    """
    Takes in a point cloud and adds it to the OccupancyGrid.
    
    Args:
        og: 2D OccupancyGrid
        points: 3xN array of xyz points
    """
    for col in eachcol(points)
        # Round x, y index and adjust for 1-indexing
        x_ind = round(Int, floor((col[1]- og.start_x) / og.cell_size)) + 1
        y_ind = round(Int, floor((col[2] - og.start_y) / og.cell_size)) + 1
        og.grid[x_ind, y_ind] = true
    end
end

function reset!(og)
    """
    Clear all cells of OccupancyGrid.
    """
    for i in eachindex(og.grid)
        og.grid[i] = false
    end
end
