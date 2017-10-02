###Wang Buzsaki Right Hand Side With Adaptation
import pyopencl as cl
import pyopencl.tools
from pyopencl import array
import numpy as np

platform = cl.get_platforms()[1]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

STV_struct = np.dtype([
    ("V", np.float32),
    ("h", np.float32),
    ("n", np.float32),
    ("M", np.float32)
    ])

STV_struct, STV_struct_c_decl = cl.tools.match_dtype_to_c_struct(
    ctx.devices[0], "STV_struct", STV_struct)

STV_struct = cl.tools.get_or_register_dtype("STV_struct", STV_struct)

prg = cl.Program(ctx, STV_struct_c_decl + """
    __kernel void BW_RHS(__global STV_struct *a)
    {
    int i = get_global_id(0);
    float bam = -0.1 * (a[i].V + 35.0);
    float a_m = bam / (exp(bam) - 1.);
    float b_m = 4.0 * exp(-(a[i].V + 60.0)/18.0);
    float m_inf = a_m / (a_m + b_m);
    float a_h = 0.07 * exp(-(a[i].V + 58.0)/20.0);
    float b_h = 1.0 / (1.0 + exp(-0.1 * (a[i].V + 28.0)));
    float ban = a[i].V + 34.0;
    float a_n = -0.01 * ban / (exp(-0.1 * ban) - 1.0);
    float b_n = 0.125 * exp(-(a[i].V + 44.0)/80.0);
    float M_inf = 1.0 / (1.0 + exp(-(a[i].V - (-10.0))/10.0));
    float I_Na = 35.0 * (pown(m_inf, 3)) * a[i].h * (a[i].V - 55.0);
    float I_K = 9.0 * (pown(a[i].n, 4)) * (a[i].V + 90.0);
    float I_L = 0.1 * (a[i].V + 65.0);
    float I_M = 1.0 * a[i].M * (a[i].V + 90.0);
    a[i].V = 0.4 - I_M - I_Na - I_K - I_L;
    a[i].h = 5.0 * (a_h * (1.0 - a[i].h) - (b_h * a[i].h));
    a[i].n = 5.0 * (a_n * (1.0 - a[i].n) - (b_n * a[i].n));
    a[i].M = (M_inf - a[i].M) / 2000.0;
    }
    """).build()


def PY_BW_RHS_ADAPT(stvars, args):
    stvars = stvars.reshape(4, args[2])
    #Sodium
    bam = -0.1*(stvars[0] + 35.)
    a_m = bam/(np.exp(bam)-1.)
    b_m = 4.*np.exp(-(stvars[0] + 60.)/18.)
    m_inf = a_m/(a_m + b_m)
    a_h = 0.07*np.exp(-(stvars[0] + 58.)/20.)
    b_h = 1./(1. + np.exp(-0.1*(stvars[0] + 28.)))
    #Potassium
    ban = stvars[0] + 34.
    a_n = -0.01*ban/(np.exp(-0.1*ban) - 1.)
    b_n = 0.125*np.exp(-(stvars[0] + 44.)/80.)
    #M-current
    M_inf = 1./(1 + np.exp(-(stvars[0] - (-10))/10.)) #v_half set to -10; Slope set to 10.; may need adjustment
    #Currents
    I_Na = 35. * (m_inf**3.) * stvars[1] * (stvars[0] - 55.)
    I_K = 9. * (stvars[2]**4.) * (stvars[0] + 90.)
    I_L = 0.1*(stvars[0] + 65.)
    I_M = args[3] * stvars[4] * (stvars[0] + 90.)
    #derivatives; all currents negative by convention
    dVdt = args[0] - I_M - I_syn - I_Na - I_K - I_L
    dhdt = 5. * (a_h*(1. - stvars[1]) - (b_h*stvars[1]))
    dndt = 5. * (a_n*(1. - stvars[2]) - (b_n*stvars[2]))
    dmdt = (M_inf - stvars[4])/2000. #tau_a~O(1s)

    return np.concatenate([dVdt, dhdt, dndt, dmdt])

def PY_BW_RHS_ADAPT_1(stvars):
    #Sodium
    bam = -0.1*(stvars[0] + 35.)
    a_m = bam/(np.exp(bam)-1.)
    b_m = 4.*np.exp(-(stvars[0] + 60.)/18.)
    m_inf = a_m/(a_m + b_m)
    a_h = 0.07*np.exp(-(stvars[0] + 58.)/20.)
    b_h = 1./(1. + np.exp(-0.1*(stvars[0] + 28.)))
    #Potassium
    ban = stvars[0] + 34.
    a_n = -0.01*ban/(np.exp(-0.1*ban) - 1.)
    b_n = 0.125*np.exp(-(stvars[0] + 44.)/80.)
    #M-current
    M_inf = 1./(1 + np.exp(-(stvars[0] - (-10))/10.)) #v_half set to -10; Slope set to 10.; may need adjustment
    #Currents
    I_Na = 35. * (m_inf**3.) * stvars[1] * (stvars[0] - 55.)
    I_K = 9. * (stvars[2]**4.) * (stvars[0] + 90.)
    I_L = 0.1*(stvars[0] + 65.)
    I_M = 1. * stvars[3] * (stvars[0] + 90.)
    #derivatives; all currents negative by convention
    dVdt = .4 - I_M - I_Na - I_K - I_L
    dhdt = 5. * (a_h*(1. - stvars[1]) - (b_h*stvars[1]))
    dndt = 5. * (a_n*(1. - stvars[2]) - (b_n*stvars[2]))
    dmdt = (M_inf - stvars[3])/2000. #tau_a~O(1s)
    return np.array([dVdt, dhdt, dndt, dmdt])

h = 0.01
time = np.arange(0, 400, h)
InitialValues = np.zeros(4)
InitialValues[0] = np.random.uniform(-75, -55)
InitialValues[1] = 1.
InitialValues[2] = 1.

stv_h = InitialValues
stv_g = cl.array.to_device(queue, stv_h)
for i in range(len(time)):
    #store stv_hist
    stv_h = method(RHS, stv_h, stv_g, prg, queue, h, args)

#that's all

def method(RHS, stv_h, stv_g, prg, h, queue, h, args):
    stv_h = RHS(stv_h, prg, queue)
    return stv_host

def BW_RHS_ADAPT_GPU(stv_h, stv_g, prg, queue):
    #set up stvars in struct form
    stv_host["V"] = stv_h[0,:]
    stv_host["h"] = stv_h[1,:]
    stv_host["n"] = stv_h[2,:]
    stv_host["M"] = stv_h[3,:]
    #transfer state variables to GPU
    stv_h = cl.array.to_device(queue, stv_host)
    #compute derivatives on GPU
    evt = prg.BW_RHS(queue, stv_h.shape, None, stv_h.data)
    return stv_h

def BW_RHS_ADAPT_GPU_NP(stv_h, args):
    #set up stvars in struct form
    args[2]["V"] = stv_h[0]
    args[2]["h"] = stv_h[1]
    args[2]["n"] = stv_h[2]
    args[2]["M"] = stv_h[3]
    #transfer state variables to GPU
    print("stv_h: ", stv_h)
    print("\n")
    print("args[2] = ary_host: ", args[2])
    print("\n")
    stv_h = cl.array.to_device(args[0], args[2])
    print("stv_h after transfer: ", stv_h)
    #compute derivatives on GPU
    evt = args[1].BW_RHS(args[0], stv_h.shape, None, stv_h.data)
    print("stv_h after GPU computing: ", stv_h)
    stv_hf = np.zeros(4)
    for i in range(4):
        stv_hf[i] = stv_h[0][i]
    print("Made it to the end")
    return stv_hf

def euler(RHS, stvars, h):
    stdevs = RHS(stvars)
    return stvars + h*stdevs

def euler_args(RHS, stvars, h, args):
    stdevs = RHS(stvars, args)
    print
    return stvars + h*stdevs



v_hist_py = np.zeros(len(time))
stvars = InitialValues
for i in range(len(time)):
    v_hist_py[i] = stvars[0]
    stvars = euler(PY_BW_RHS_ADAPT_1, stvars, h)


# v_hist_cl = np.zeros(len(time))
# for i in range(len(time)):
#     v_h


ary_host = np.empty(1, STV_struct)
ary_host["V"].fill(-75)
ary_host["h"].fill(1)
ary_host["n"] = 1
ary_host["M"] = 0.0
v_hist_cl = np.zeros(len(time))

args = [queue, prg, ary_host]
x = euler_args(BW_RHS_ADAPT_GPU_NP, stvars, h, args)









"""
Need to call numerical method at every step
Numerical method calls RHS at every step
Context and program built in advance, so only need queue, and stvars at each step?




















#


#
