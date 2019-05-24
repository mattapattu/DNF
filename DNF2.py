import spynnaker8 as p
import sys
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import time
from collections import deque


runtime = 240000
p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 160)

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 0.1,
                   'tau_syn_E': 40.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }


cell_params_lif1 = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 0.1,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

populations = list()
projections = list()

delay = 1.0

loopConnections = list()
for i in range(0, 60):
    singleConnection = ((i, 0, 1, delay))
    loopConnections.append(singleConnection)

for i in range(60,150):
    singleConnection = ((i, 1, 1, delay))
    loopConnections.append(singleConnection)

for i in range(150,160):
    singleConnection = ((i, 2, 1, delay))
    loopConnections.append(singleConnection)

for i in range(160,170):
    singleConnection = ((i, 3, 1, delay))
    loopConnections.append(singleConnection)

for i in range(170,260):
    singleConnection = ((i, 4, 1, delay))
    loopConnections.append(singleConnection)

for i in range(260, 320):
    singleConnection = ((i, 5, 1, delay))
    loopConnections.append(singleConnection)


populations.append(
    p.Population(320, p.IF_curr_exp(**cell_params_lif), label='pop_1'))
populations.append(
    p.Population(6, p.IF_curr_exp(**cell_params_lif1), label='pop_2'))
populations.append(
    p.Population(320, p.external_devices.SpikeInjector(label='inputSpikes_1',database_notify_port_num=19990)))

projections.append(p.Projection(
    populations[0], populations[1], p.FromListConnector(loopConnections),
    ))
projections.append(p.Projection(
    populations[2], populations[0], p.OneToOneConnector(),
    p.StaticSynapse(weight=1, delay=1)))


populations[0].record(['spikes'])
populations[1].record(['spikes'])


p.external_devices.activate_live_output_for(populations[1])

live_connection = p.external_devices.SpynnakerLiveSpikesConnection(
    receive_labels=['pop_2'])

neuron0 = 0
neuron1 = 0
neuron2 = 0
neuron3 = 0
neuron4 = 0
neuron5 = 0 
lastspiketime = 0
current_milli_time = lambda: int(round(time.time() * 1000))

def receive_spikes_1(label, time, neuron_ids):
   print("Received spikes from population {}, neurons {} at time {}".format(
        label, neuron_ids, time))
   global neuron0,neuron1,neuron2,neuron3,neuron4,neuron5,lastspiketime
   t = current_milli_time()
   if t >= lastspiketime + 100:
        printRatesAndResetCounter()
   lastspiketime = t
   if 0 in neuron_ids :
         neuron0 += 1
   elif 1 in neuron_ids :
         neuron1 += 1
   elif 2 in neuron_ids :
         neuron2 += 1
   elif 3 in neuron_ids :
         neuron3 += 1
   elif 4 in neuron_ids :
         neuron4 += 1
   elif 5 in neuron_ids :
         neuron5 += 1
   if t >= lastspiketime + 100:
        printRatesAndResetCounter()

def printRatesAndResetCounter():
   global neuron0,neuron1,neuron2,neuron3,neuron4,neuron5,lastspiketime
   neuron0rate = neuron0/float(2)
   neuron1rate = neuron1/float(2)
   neuron2rate = neuron2/float(2)
   neuron3rate = neuron3/float(2)
   neuron4rate = neuron4/float(2)
   neuron5rate = neuron5/float(2)
   print "neuron0 rate = ", neuron0rate
   print "neuron1 rate = ", neuron1rate
   print "neuron2 rate = ", neuron2rate
   print "neuron3 rate = ", neuron3rate
   print "neuron4 rate = ", neuron4rate
   print "neuron5 rate = ", neuron5rate
   neuron0 = 0 
   neuron1 = 0
   neuron2 = 0
   neuron3 = 0
   neuron4 = 0
   neuron5 = 0
   
live_connection.add_receive_callback('pop_2', receive_spikes_1)     
p.run(runtime)
 

#print projections[0].get(["weight", "delay"], format="list")

#print projections[1].get(["weight", "delay"], format="list")


 ### get data (could be done as one, but can be done bit by bit as well)
spikes_1 = populations[0].get_data('spikes')
spikes_2 = populations[1].get_data('spikes')
line_properties = [{'color': 'red', 'markersize': 2},
                   {'color': 'blue', 'markersize': 2}]
Figure(
   # raster plot of the presynaptic neuron spike times
   Panel(spikes_1.segments[0].spiketrains,
         yticks=True,xticks=True, line_properties=line_properties, xlim=(0, runtime)),
   Panel(spikes_2.segments[0].spiketrains,
         yticks=True,xticks=True, line_properties=line_properties, xlim=(0, runtime)),
   title="DNF",
   annotations="Simulated with {}".format(p.name())
)

plt.show()
p.end()
