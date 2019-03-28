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

p.setup(timestep=1.0)


end_time = 30
SpikeArray = {
    'spike_times':
    [[0,1,2,3,5,6,7,8,9,10],
     [],
     [],
     []]
    }
    

spikeinput=p.Population(3,p.SpikeSourceArray, SpikeArray,label="input")
motor_pop1=p.Population(2,p.IF_curr_exp,{},label="motor neurons")
conn_list = [(0,0,5,1),(1,0,5,1),(1,1,5,1),(2,1,5,1)]
spike_motor_projection=p.Projection(spikeinput,motor_pop1,p.OneToOneConnector(weights=5.0,delays=1),target="excitatory")



pop_1.record()
pop_1.record_v()

motor_pop.record()
motor_pop.record_v()

p.run(end_time)

import pylab

def plotMembranePotential(v,id) :
	time=[i[1] for i in v if i[0]==id]
	membrane_voltage=[i[2]for i in v if i[0]==id]
	pylab.plot(time,membrane_voltage)
	pylab.xlabel("Time(ms)")
	pylab.ylabel("MembraneVoltage of neuron "+ str(id))
	pylab.axis([0,end_time,-75,-45])
	pylab.show()


def plotSpikeTimes(spikes,id) :
	spike_time=[i[1] for i in spikes if i[0]==id]
	spike_id=[i[0] for i in spikes if i[0]==id ]
	pylab.plot(spike_time,spike_id,".")
	pylab.xlabel("Time(ms)")
	pylab.ylabel("NeuronID ="+ str(id))
	pylab.axis([0,end_time,0,2])
	pylab.show()


print pop_1.get_v()
#print "###########################"
print pop_1.getSpikes()
#plotMembranePotential(pop_1.get_v(),0)
#plotSpikeTimes(pop_1.getSpikes(),0)

#plotMembranePotential(pop_1.get_v(),1)
#plotSpikeTimes(pop_1.getSpikes(),1)

#plotMembranePotential(pop_1.get_v(),2)
#plotSpikeTimes(pop_1.getSpikes(),2)

#print motor_pop.get_v()
#print motor_pop.getSpikes()

plotMembranePotential(motor_pop.get_v(),0)
plotSpikeTimes(motor_pop.getSpikes(),0)

plotMembranePotential(motor_pop.get_v(),1)
plotSpikeTimes(motor_pop.getSpikes(),1)

def rateCoding(spikes) :
	default = 1.0 #set to a default speed when first spike comes. 

	### depending on the rate every 10 milli sec, increase or stay at default.
	spike_rm = PD.rolling_mean(spikes, 10)	
	print spike_rm
