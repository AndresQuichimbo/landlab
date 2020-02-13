# -*- coding: utf-8 -*-
"""
Created on Jul 2018

@author: andresquichimbo@gmail.com

"""


import pdb
from landlab import Component,FieldError
import numpy as np

class TransmissionLosses(Component):
    """
    Calculate transmission losses
    
    Landlab component that implements a two-dimensional
    
    Construction:
    
        TransmissionLosses(grid, precip_rate=1.0, 
                                 precip_duration=1.0, 
                                 infilt_rate=0.0,
                                 roughness=0.01, **kwds)
    
    Parameters
    ----------
    grid : ModelGrid
        A Landlab grid object.
    precip_rate : float, optional (defaults to 1 mm/hr)
        Precipitation rate, mm/hr
    precip_duration : float, optional (defaults to 1 hour)
        Duration of precipitation, hours
    infilt_rate : float, optional (defaults to 0)
        Maximum rate of infiltration, mm/hr
    roughnes : float, defaults to 0.01
        Manning roughness coefficient, s/m^1/3
        
    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> rg = RasterModelGrid((4, 5), 10.0)
    >>> kw = KinwaveOverlandFlowModel(rg)
    >>> kw.vel_coef
    100.0
    >>> rg.at_node['surface_water__depth']
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.])    
    """

    _name = 'KinwaveOverlandFlowModela'

    _input_var_names = (
        'topographic__elevation',
        'conductivity',
        'water_table',
    )

    _output_var_names = (
        'topographic__gradient',
        'surface_water__depth',
        'water__velocity',
        'water__specific_discharge',
        'transmission_losses',
        'conductivity',
    )

    _var_units = {
        'topographic__elevation' : 'm',
        'topographic__slope' : 'm/m',
        'surface_water__depth' : 'm',
        'water__velocity' : 'm/s',
        'water__specific_discharge' : 'm2/s',
        'transmission_losses':'m3/s',
        'conductivity':'m3/s2',
        'water_table':'m',
    }

    _var_mapping = {
        'topographic__elevation' : 'node',
        'topographic__gradient' : 'link',
        'surface_water__depth' : 'node',
        'water__velocity' : 'link',
        'water__specific_discharge' : 'link',
        'transmission_losses':'node',
        'conductivity':'node',
        'water_table':'node',
    }

    _var_doc = {
        'topographic__elevation':
            'elevation of the ground surface relative to some datum',
        'topographic__gradient':
            'gradient of the ground surface',
        'surface_water__depth':
            'depth of water',
        'water__velocity':
            'flow velocity component in the direction of the link',
        'water__specific_discharge':
            'flow discharge component in the direction of the link',    
        'transmission_losses':
            'transmission losses from ephemeral streams',
        'conductivity':
            'conductance of the river',
        'water_table':
            'water table above a datum',
    }

    def __init__(self, grid, precip_rate=1.0, precip_duration=1.0, 
                 infilt_rate=0.0, roughness=0.01,conductivity=0.0, transmission_losses=0.0, **kwds):
        """Initialize the KinwaveOverlandFlowModel.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        precip_rate : float, optional (defaults to 1 mm/hr)
            Precipitation rate, mm/hr
        precip_duration : float, optional (defaults to 1 hour)
            Duration of precipitation, hours
        infilt_rate : float, optional (defaults to 0)
            Maximum rate of infiltration, mm/hr
        roughnes : float, defaults to 0.01
            Manning roughness coefficient, s/m^1/3
        conductivity : float, defaults to 0.001
            C=(k/t) *lw, m3/s
        """

        # Store grid and parameters and do unit conversion
        self._grid = grid
        self._bc_set_code = self.grid.bc_set_code
        self.precip = precip_rate / 3600000.0 # convert to m/s
        self.precip_duration = precip_duration * 3600.0  # h->s
        self.infilt = infilt_rate / 3600000.0 # convert to m/s
        self.vel_coef = 1.0 / roughness  # do division now to save time
        self.conduct=conductivity
        self.trans_losses=transmission_losses

        # Create fields...
        #   Elevation
        if 'topographic__elevation' in grid.at_node:
            self.elev = grid.at_node['topographic__elevation']
        else:
            raise FieldError(
                'A topography is required as a component input!')
        #   Precipitation
        if 'precip_rate' in grid.at_node:
            self.precip = grid.at_node['precip_rate']
        else:
            self.precip = grid.add_zeros('node', 'precip_rate')
        #   infiltration
        if 'infilt_rate' in grid.at_node:
            self.infilt = grid.at_node['infilt_rate']
        else:
            self.infilt = grid.add_zeros('node', 'infilt_rate')
        #   Water table
        if 'water_table' in grid.at_node:
            self.wtable = grid.at_node['water_table']
        else:
            self.wtable = grid.add_zeros('node', 'water_table')
        #  Water depth
        if 'surface_water__depth' in grid.at_node:
            self.depth = grid.at_node['surface_water__depth']
        else:
            self.depth = grid.add_zeros('node', 'surface_water__depth')
        #   Slope
        if 'topographic__gradient' in grid.at_link:
            self.slope = grid.at_link['topographic__gradient']
        else:
            self.slope = grid.add_zeros('link', 'topographic__gradient')
        #  Velocity
        if 'water__velocity' in grid.at_link:
            self.vel = grid.at_link['water__velocity']
        else:
            self.vel = grid.add_zeros('link', 'water__velocity')
        #  Discharge
        if 'water__specific_discharge' in grid.at_link:
            self.disch = grid.at_link['water__specific_discharge']
        else:
            self.disch = grid.add_zeros('link',
                                        'water__specific_discharge')
        #  Transmission losses
        if 'transmission_losses' in grid.at_node:
            self.trans_losses = grid.at_node['transmission_losses']
        else:
            self.trans_losses = grid.add_zeros('node',
                                        'transmission_losses')
        # Conductivity
        if 'conductivity' in grid.at_node:
            self.conduct = grid.at_node['conductivity']
        else:
            self.conduct = grid.add_zeros('node','conductivity')
			
		# Initial soil moisture
        if 'moisture' in grid.at_node:
            self.theta = grid.at_node['moisture']
        else:
            self.theta = grid.add_zeros('node','moisture')+0.35

        # Calculate the ground-surface slope
        self.slope[self._grid.active_links] = \
            self._grid.calc_grad_at_link(self.elev)[self._grid.active_links]
        self.sqrt_slope = np.sqrt( self.slope )
        self.sign_slope = np.sign( self.slope )

    def updated_boundary_conditions(self):
        """Call if boundary conditions are updated.
        """
        self.slope[self.grid.active_links] = \
            self.grid.calc_grad_at_link(self.elev)[self.grid.active_links]
        self.sqrt_slope = np.sqrt(self.slope)
        self.sign_slope = np.sign( self.slope )

    def run_one_step(self,dt, **kwds):
        """Calculate water flow for a time period `dt`.
        """

        #if self._bc_set_code != self.grid.bc_set_code:
        #    self.updated_boundary_conditions()
        #    self._bc_set_code = self.grid.bc_set_code

        # Calculate water depth at links
        #H_link = self._grid.map_value_at_max_node_to_link(
        #        'topographic__elevation', 'surface_water__depth')

        # Calculate velocity
        #self.vel = -self.sign_slope * self.vel_coef * H_link**0.66667 \
        #            * self.sqrt_slope

        # Calculate discharge
        #self.disch = H_link * self.vel

        # Flux divergence
        #dqda = self._grid.calc_flux_div_at_node(self.disch)

        # Rate of change of water depth
       # if current_time < self.precip_duration:
        #    ppt = self.precip
        #else:
        #    ppt = 0.0
        #dHdt = ppt - self.infilt - dqda
        #ro = ppt - self.infilt - dqda
        
        dxdy=30.0*30.0
        q_river=self.depth*dxdy
        
        #C_bed=self.conduct[self._grid.core_nodes]
        #pdb.set_trace()
        tl=self.conduct*((1/6*np.sqrt(1/2*9.81)*np.abs(q_river))**2/3)
        
        #dHdt=ro-tl/10000
        #pdb.set_trace()
        self.trans_losses[self._grid.core_nodes] = tl[self._grid.core_nodes]*dt/dxdy
        #pdb.set_trace()
        ne=0.30
        Ha=15.0
        K=self.conduct[self._grid.core_nodes]
        h=tl[self._grid.core_nodes]
        z=0.0
        L0=self.theta[self._grid.core_nodes]
        L=rk5_FR(L0,h,Ha,z,K,tl[self._grid.core_nodes],ne,dt)
        #L=rk5_FR(L0,h,Ha,z,K,TL,ne,dt)
        self.theta[self._grid.core_nodes]=L
        #Recharge=(L/Lmax)*(L/ne)
        #self.at_nodes[self.trans_losses] +=tl
        #self.trans_losses[self._grid.core_nodes] += tl[self._grid.core_nodes]
        # Update water depth
        self.depth[self._grid.core_nodes] -= self.trans_losses[self._grid.core_nodes]
        

        # Somewhat crude numerical hack: prevent negative water depth
        self.depth[np.where(self.depth < 0.0)[0]] = 0.000001

	#-----------------------------------------------------------------
	
	#function [f TL R]=fun_FRch(L0,y,Ha,z,K,Tif,CKTL,CKUZ)
def fun_FRch(L0,h,Ha,z,K,TL,ne):
	####### Lineal reservoir for UZ zone ######
	## Paramenters
	## y = river stage
	## z = channel bottom
	## K = Hydraulic conductivity
	## L0= Initial conditions
	## Tif = Saturation decay 0 <= Tif < 1
	## CKTL =  Saturation decay 0 < CKTL < 1
	## CKUZ =  Rezise the unsaturated zone
	## Initial conditions for the reservoir
    L = L0
    Lmax=ne*Ha
    aux_1=Lmax-L
    aux_1[np.where(aux_1 >= 0.0 )]= 0.0
    aux_1[np.where(aux_1 < 0.0 )]= 1.0
    aux_3=(L/Lmax)*(L/ne)*aux_1
    aux_2=L-Lmax
    aux_2[np.where(aux_2 >= 0.0 )]= 0.0
    aux_2[np.where(aux_2 < 0.0 )]= 1.0
    aux_4=aux_2*K*(h+Ha)/Ha
    R=aux_3+aux_4
	#if L>=Lmax:			
	#	# Transmission losses 			
	#	R=K*(h+Ha)/Ha		
	#else: #check this for discharges
	#	#TL = 0
	#	R=(L/Lmax)*(L/ne)
	#
    f=TL-R
    return (f,TL,R)



def rk5_FR(L0,h,Ha,z,K,TL,ne,dt):
    
	#function [TLt Rt dldt]=rk5(L0,y,Ha,z,K,Tif,CKTL,CKUZ)
	#N=1 #len(y); # N time steps
	#dt = 1
	#dldt=[]
	#TLt=[]
	#Rt=[]
	#RUNGE KUTTA METHOD FOR SOLVING EDO
	#L=[]
    z=0.0
	#L.append([L0]) # Initial condition of the reservoir
	#for j in range(0,N): # N time steps
    k1=fun_FRch(L0,h,Ha,z,K,TL,ne) # K1 estimation of RK5 method
    #pdb.set_trace()
    k2=fun_FRch(L0+1.0/2.0*dt*k1[:][0],h,Ha,z,K,TL,ne) # K2 estimation of RK5 method
    k3=fun_FRch(L0+1.0/2.0*dt*k2[:][0],h,Ha,z,K,TL,ne) # K3 estimation of RK5 method
    k4=fun_FRch(L0+dt*k3[:][0],h,Ha,z,K,TL,ne) # K4 estimation of RK5 method
    L=L0+dt*(k1[:][0]+2.0*k2[:][0]+2.0*k3[:][0]+k4[:][0])/6.0# solution
    return (L)