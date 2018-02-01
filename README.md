# GPU
Here I toyed with using PyOpenCL to do my Hodgkin-Huxley type network simulations on a GPU. 
This is potentially helpful because at each time step you can solve currents for each neuron in parallel. 
To do this you have to transfer data to and from the CPU at each time step. I was able to simulate uncoupled neurons in parallel.

To Do:
Figure out how to store weights matrix in Local Memory (or what the optimal arrangement for storing my data is)
Simulate networks no just uncoupled neurons.

Issues:
Getting a weird issue with lockfiles...seems to slow my code down to the point of not being functional.

Strong synapses seem to make the system stiff, which requires an adaptive solver. When I interface with
the SciPy implementation of VODE I suspect that having to make multiple function calls at each time step
means transfering data back and forth multiple times per time step, which slows us down.
