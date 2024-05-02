from linpde_gp.domains import TensorProductDomain, Box
import numpy as np

def get_grid_from_resolution(box_domain: Box, resolution: int | list[int]):
    if not isinstance(resolution, list):
        resolution = [resolution] * box_domain.dimension
    assert len(resolution) == box_domain.dimension

    return TensorProductDomain.from_endpoints(
        *(
            np.linspace(boundaries[0], boundaries[1], res + 1)
            for boundaries, res in zip(box_domain, resolution)
        )
    )

def get_grid_from_depth(box_domain: Box, depth: int):
    return get_grid_from_resolution(box_domain, 2 ** depth)