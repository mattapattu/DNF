#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:28:53 2019

@author: ajames
"""

try:
    import pyNN.spiNNaker as p
except Exception as e:
    import spynnaker.pyNN as p

import pylab 
import matplotlib.pyplot as plt
import numpy as np   
import operator

p.setup(timestep=1.0)


end_time = 20
dnfNeuronActivations = np.random.random_sample(60)
print(dnfNeuronActivations)
spike_times = list()
for x in dnfNeuronActivations:
    new_list = [1/x]
    spike_times.append(new_list)

SpikeArray = {
    'spike_times': spike_times
    }

id,first_spike = max(enumerate(dnfNeuronActivations), key=operator.itemgetter(1))   
print(spike_times)
print("First spike is from %i at %f" % (id,first_spike))
 
dnfNeuronCount = 60
weightStep = 5
weightdiff = 5
initialWeight = 10

spikeinput=p.Population(dnfNeuronCount,p.SpikeSourceArray, SpikeArray,label="input")

dnf_pop=p.Population(dnfNeuronCount,p.IF_curr_exp,cellparams={'v_rest': 0,'v_thresh': 1.0, 'tau_m': 10.0},label = "DNF dir neurons")
dnf_inhibition_proj = p.Projection(dnf_pop,dnf_pop,p.AllToAllConnector(allow_self_connections=False,weights=1.0,delays=0),target="inhibitory")
spikesToDnf_proj=p.Projection(spikeinput,dnf_pop,p.OneToOneConnector(weights=1.0,delays=0),target="excitatory")

motor_left=p.Population(1,p.IF_curr_exp,{},label="motor neuron left")
motor_right=p.Population(1,p.IF_curr_exp,{},label="motor neuron right")


weights1 = np.arange(initialWeight,initialWeight+((dnfNeuronCount/2)*weightStep), weightStep)
weights1_flip = weights1[::-1]
weights2 = np.arange(initialWeight,initialWeight+((dnfNeuronCount/2)*weightStep), weightStep)
dnfToRightNeuron = np.concatenate((weights1_flip,weights2-5))
dnfToLeftNeuron =  np.concatenate((weights1_flip-5,weights2))
print(dnfToRightNeuron)
print(dnfToLeftNeuron)
dnfToRightNeuron_proj = p.Projection(dnf_pop,motor_right,p.AllToAllConnector(weights=dnfToRightNeuron, delays=1),target="excitatory")
dnfToLeftNeuron_proj = p.Projection(dnf_pop,motor_left,p.AllToAllConnector(weights=dnfToLeftNeuron, delays=1),target="excitatory")

spikeinput.record()

dnf_pop.record()
dnf_pop.record_v()

motor_right.record()
motor_right.record_v()

motor_left.record()
motor_left.record_v()

p.run(20)

def plotMembranePotential(v,id) :
	time=[i[1] for i in v if i[0]==id]
	membrane_voltage=[i[2]for i in v if i[0]==id]
	pylab.plot(membrane_voltage,time)
	pylab.xlabel("Time(ms)")
	pylab.ylabel("MembraneVoltage of neuron "+ str(id))
	pylab.axis([0,end_time,-1000,20])
	pylab.show()


def plotSpikeTimes(spikes,id) :
	spike_time=[i[1] for i in spikes if i[0]==id]
	spike_id=[i[0] for i in spikes if i[0]==id ]
	pylab.plot(spike_time,spike_id,".")
	pylab.xlabel("Time(ms)")
	pylab.ylabel("NeuronID ="+ str(id))
	pylab.axis([0,end_time,0,2])
	pylab.show()

mem = dnf_pop.get_v(gather=True, compatible_output=True)
spik = dnf_pop.getSpikes()

time = [i[1] for i in mem if i[0] == id]
membrane_voltage = [i[2] for i in mem if i[0] == id]
pylab.plot(time, membrane_voltage)
pylab.xlabel("Time (ms)")
pylab.ylabel("Membrane Voltage id")
pylab.axis([0, 20, -500, 20])
pylab.show()

pylab.plot([i[1] for i in spik], [i[0] for i in spik], ".")
pylab.xlabel("Time (ms)")
pylab.ylabel("Neuron ID")
pylab.axis([0, 20, -1, 60 + 1])
pylab.show()


spike_counts = spikeinput.get_spike_counts()
for id in sorted(spike_counts):
    print(id, spike_counts[id])

spike_counts = dnf_pop.get_spike_counts()
for id in sorted(spike_counts):
    print(id, spike_counts[id])


#plotMembranePotential(mem,51.0)
#plotSpikeTimes(pop_1.getSpikes(),1)

#plotMembranePotential(pop_1.get_v(),2)
#plotSpikeTimes(pop_1.getSpikes(),2)

#print motor_pop.get_v()
#print motor_pop.getSpikes()

#plotMembranePotential(motor_left.get_v(),0)
#plotSpikeTimes(motor_left.getSpikes(),0)

#plotMembranePotential(motor_right.get_v(),0)
#plotSpikeTimes(motor_right.getSpikes(),0)
