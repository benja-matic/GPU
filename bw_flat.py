###solve wang-buszaki with adaptation without any struct business
import pyopencl as cl
import pyopencl.tools
from pyopencl import array
import numpy as np

#set context to my GPU, set up queue to it
platform = cl.get_platforms()[1]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)


###read in state variables from __constant Buffer for each neuron
###calculate slopes of all variables
###write slopes to global memory
prg_p = cl.Program(ctx, """
    #define N_VAR 4
    __kernel void BW_RHS(__constant float *stvars_g_ptr, __global float *stdrvs_g_ptr)
    {
    int i = get_global_id(0);
    int N = get_global_size(0);
    float stvars[N_VAR];
    float stdrvs[N_VAR];

    for(int j=0; j < N_VAR; ++j){
        stvars[j] = stvars_g_ptr[i + j*N];
    }
    float bam = -0.1 * (stvars[0] + 35.0);
    float a_m = bam / (exp(bam) - 1.);
    float b_m = 4.0 * exp(-(stvars[0] + 60.0)/18.0);
    float m_inf = a_m / (a_m + b_m);
    float a_h = 0.07 * exp(-(stvars[0] + 58.0)/20.0);
    float b_h = 1.0 / (1.0 + exp(-0.1 * (stvars[0] + 28.0)));
    float ban = stvars[0] + 34.0;
    float a_n = -0.01 * ban / (exp(-0.1 * ban) - 1.0);
    float b_n = 0.125 * exp(-(stvars[0] + 44.0)/80.0);
    float M_inf = 1.0 / (1.0 + exp(-(stvars[0] - (-10.0))/10.0));
    float I_Na = 35.0 * (pown(m_inf, 3)) * stvars[1] * (stvars[0] - 55.0);
    float I_K = 9.0 * (pown(stvars[2], 4)) * (stvars[0] + 90.0);
    float I_L = 0.1 * (stvars[0] + 65.0);
    float I_M = 1.0 * stvars[3] * (stvars[0] + 90.0);
    stdrvs[0] = 0.4 - I_M - I_Na - I_K - I_L;
    stdrvs[1] = 5.0 * (a_h * (1.0 - stvars[1]) - (b_h * stvars[1]));
    stdrvs[2] = 5.0 * (a_n * (1.0 - stvars[2]) - (b_n * stvars[2]));
    stdrvs[3] = (M_inf - stvars[3]) / 2000.0;

    for(int j=0; j < N_VAR; ++j){
        stdrvs_g_ptr[i + j*N] = stdrvs[j];
    }

    }
    """).build()

#
prg_p2 = cl.Program(ctx, """
    #define N_VAR 4
    __kernel void BW_RHS(__constant float *stvars_g_ptr, __global float *stdrvs_g_ptr)
    {
    int i = get_global_id(0);
    int N = get_global_size(0);
    float stvars[N_VAR];
    float stdrvs[N_VAR];

    for(int j=0; j < N_VAR; ++j){
        stvars[j] = stvars_g_ptr[i + j*N];
    }

    for(int j=0; j < N_VAR; ++j){
        stdrvs_g_ptr[i + j*N] = stvars[j];
    }
    stdrvs_g_ptr[0] = N_VAR;
    }
    """).build()

#host stuff
N = 32
h = 0.01
time = np.arange(0, 400, h)
InitialValues = np.zeros(N*4)
InitialValues[:N] = np.random.uniform(-75, -55, N)
InitialValues[N:3*N] = 1.
InitialValues[N*3:] = np.random.uniform(0, 0.05, N)

def BWA_GPU_RHS(stvars_h, args):
    stvars_g = cl.Buffer(args[0], cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=stvars_h) #pass state variables to GPU (constant mem)
    stdrvs_h = np.zeros(args[3]) #CPU result buffer
    args[2].BW_RHS(args[1], args[3], None, stvars_g, args[4]) #run the kernel
    cl.enqueue_copy(args[1], stdrvs_h, args[4]) #pass result from GPU back to CPU
    return stdrvs_h

stdrvs_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, InitialValues.nbytes) #GPU result buffer
args = [ctx, queue, prg_p2, (long(N*4),), stdrvs_g]
I2 = BWA_GPU_RHS(InitialValues, args)

























#
