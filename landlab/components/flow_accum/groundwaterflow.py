import numpy as np

from landlab import Component, FieldError
from landlab.grid.structured_quad import links
from landlab.utils.decorators import use_file_name_or_kwds

class GroundwaterFlow(Component):

    _name = "GroundwaterFlow"

    _cite_as = """, title={GroundwaterFlow component
        }
    """

    _input_var_names = ("water_table__depth",
		"topographic__elevation",
		"bottom_elevation",
		"recharge",
		"hydraulic_conductivity",
		"Specific_yield")

    _output_var_names = (
        "water_table__depth",
    )

    _var_units = {
        "water_table__depth": "m",
		"recharge": "m",
        "bottom_elevation": "m",
        "topographic__elevation": "m",
		"hydraulic_conductivity": "m/d",
        "Specific_yield": "-",
    }

    _var_mapping = {
        "water_table__depth": "node",
		"recharge": "node",
        "bottom_elevation": "node",
		"topographic__elevation": "node",
        "hydraulic_conductivity": "link",
		"Specific_yield": "node",
        "water_surface__gradient": "link",
    }

    _var_doc = {
        "surface_water__depth": "The depth of water at each node.",
        "topographic__elevtation": "The land surface elevation.",
        "surface_water__discharge": "The discharge of water on active links.",
        "water_surface__gradient": "Downstream gradient of the water surface.",
    }
	    @use_file_name_or_kwds
    def __init__(
        self,
        grid,
        default_fixed_links=False,
        h_init=0.00001,
        alpha=0.7,
        K_n=0.03,
        g=9.81,
        recharge=0.0,
        steep_slopes=False,
        **kwds
    ):
	
	
	
	
	
	
    def calc_time_step(self):
        """Calculate time step.

        Adaptive time stepper from Bates et al., 2010 and de Almeida
        et al., 2012
        """
        self.dt = (
			0.9*0.25*self._grid.at_node["Specific_yield"]
			self._grid.dx**2
			/ np.amax(self._grid.at_node["hydraulic_conductivity"]*
			self._grid.at_node["pressure_head"]))
		)

        return self.dt