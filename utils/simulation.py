import os
import numpy as np
from tvb.simulator.backend.nb_mpr import NbMPRBackend
#from src.nb_mpr import NbMPRBackend as my_backend
from tvb.datatypes.time_series import TimeSeriesRegion
from bold import BalloonModel
import pickle as cPickle
from tvb.simulator.lab import *
from shapely.geometry import LineString


def configure_conn(conn_file, conn_speed):
    conn = connectivity.Connectivity.from_file( conn_file )
    conn.speed = np.array([conn_speed])
    np.fill_diagonal(conn.weights, 0.)
    # cutoff = np.argwhere(conn.cortical).max()
    # temp_weights = conn.weights.copy()
    # temp_weights[cutoff+1:,cutoff+1:] = 0
    # subcort = conn.weights[cutoff+1:,cutoff+1:]/np.max(conn.weights[cutoff+1:,cutoff+1:])
    # conn.weights = temp_weights
    conn.weights = conn.weights/conn.weights.max()
    # conn.weights[cutoff+1:,cutoff+1:] = subcort
    conn.configure()
    return conn


def run_nbMPR_backend(sim, **kwargs):
    backend = NbMPRBackend()
    return backend.run_sim(sim, **kwargs)


def tavg_to_bold(tavg_t, tavg_d, sim=None, tavg_period=None, connectivity=None, svar=0, decimate=2000):
    if sim is not None:
        assert len(sim.monitors) == 1
        tavg_period = sim.monitors[0].period
        connectivity = sim.connectivity
    else:
        assert tavg_period is not None and connectivity is not None
 
    tsr = TimeSeriesRegion(
        connectivity = connectivity,
        data = tavg_d[:,[svar],:,:],
        time = tavg_t,
        sample_period = tavg_period
    )
    tsr.configure()

    bold_model = BalloonModel(time_series = tsr, dt=tavg_period/1000)
    bold_data_analyzer  = bold_model.evaluate()

    bold_t = bold_data_analyzer.time[::decimate] * 1000 # to ms
    bold_d = bold_data_analyzer.data[::decimate,:]

    return bold_t, bold_d 


class MontbrioPazoRoxinReversed(models.MontbrioPazoRoxin):
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        derivative = - super( MontbrioPazoRoxinReversed, self).dfun(
                state_variables, coupling, local_coupling)
        
        return derivative


def mpr_fixed_points(mpr):
    I = mpr.I.item()
    Delta = mpr.Delta.item()
    Gamma = mpr.Gamma.item()
    eta = mpr.eta.item()
    tau = mpr.tau.item()
    J = mpr.J.item()

    rr = np.linspace(*mpr.state_variable_range['r'],100)
    VV = np.linspace(*mpr.state_variable_range['V'],100)

    V0=[]
    r0=[]
    for r in rr:
        r0.append(np.roots( [2*r, + Delta / (np.pi)] ) ) 

    for V in VV:
        V0.append(np.roots([- np.pi**2, J , V**2 + eta + I]))

    r_line = LineString([ (r,V.item()) for (r,V) in zip(rr,r0) if len(V)>0] )
    V_line1 = LineString([ (r[0],V) for (r,V) in zip(V0,VV)] )
    V_line2 = LineString([ (r[1],V) for (r,V) in zip(V0,VV)] )
    
    intersection = r_line.intersection(V_line2) 

    if intersection.geom_type == 'MultiPoint':
        down_state = intersection[0].xy
        saddle = intersection[1].xy
        if down_state[1][0] > saddle[1][0]:
            down_state, saddle = saddle, down_state
    else:
        assert(len(intersection.coords)==0)
        down_state = saddle = None
    up_state = r_line.intersection(V_line1).xy

    return down_state, saddle, up_state

def upstate_basin(mpr,saddle, up):
    mpr_back = MontbrioPazoRoxinReversed(
        I     = mpr.I,
        eta   = mpr.eta,
        J     = mpr.J,
        Delta = mpr.Delta,
    )
    ic = np.zeros(buffer_shape(mpr,1,1))
    if saddle:
        ic[0,:,0,0] = saddle[0][0], saddle[1][0]
        raw_d1 = integrate_decoupled_nodes(mpr_back, N=1, nsteps=30*100, deterministic=True, initial_conditions=ic-0.001)
        raw_d2 = integrate_decoupled_nodes(mpr_back, N=1, nsteps=30*100, deterministic=True, initial_conditions=ic+0.001)
    else:
        ic[0,:,0,0] = up[0][0]+.0, up[1][0]+0.0
        raw_d1 = integrate_decoupled_nodes(mpr_back, N=1, nsteps=30*150, deterministic=True, initial_conditions=ic-0.001)
        raw_d2 = integrate_decoupled_nodes(mpr_back, N=1, nsteps=30*150, deterministic=True, initial_conditions=ic+0.001)

    return raw_d1, raw_d2


def configure_sim(G=None, nsigma=None, stimulus=None, seed=42, conn_speed=20., dt=0.01, conn=None, period=0.1, initial_conditions=None):
    sim = simulator.Simulator(
        model=models.MontbrioPazoRoxin(
            eta   = np.r_[-5.0],
            J     = np.r_[15.],
            Delta = np.r_[1.],
        ),
        connectivity=conn,
        coupling=coupling.Linear(
            a=np.array([G])
        ),
        conduction_speed=conn_speed,
        integrator=integrators.HeunStochastic(
            dt=dt,
            noise=noise.Additive(
                nsig=np.array(
                    [nsigma,nsigma*2]
                ), 
                noise_seed=seed)
        ),
        monitors=[
            monitors.TemporalAverage(period=period) # we rescale time later, to "slow down" the system
        ],
        stimulus=stimulus,
        initial_conditions=initial_conditions
    ).configure()
    return sim


def build_stimulus(onset=10, node=None, conn=None, amp=None, duration=10):
    eqn_t = equations.PulseTrain()
    weighting = np.zeros((84,))
    weighting[node] = .9
    eqn_t.parameters['onset'] = onset
    eqn_t.parameters['T'] = 40000
    eqn_t.parameters['tau'] = duration
    eqn_t.parameters['amp'] = amp
    stimulus = patterns.StimuliRegion(
        temporal=eqn_t,
        connectivity=conn,
        weight=weighting
    )
    return stimulus
