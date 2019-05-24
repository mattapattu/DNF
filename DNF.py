import spynnaker8 as p
import sys
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import time
from collections import deque


runtime = 200
timestep = 0.01
delay = 0.01
p.setup(timestep=timestep, min_delay=delay, max_delay=delay)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 160)

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 1.0,
                   'tau_refrac': 0.1,
                   'tau_syn_E': 2.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }


cell_params_lif1 = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 1.5,
                   'tau_refrac': 0.1,
                   'tau_syn_E': 2.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

populations = list()
projections = list()


loopConnections = list()
for i in range(0, 100):
    singleConnection = ((i, 0, 1, delay))
    loopConnections.append(singleConnection)
    singleConnection = ((i, 1, 60, delay))
    loopConnections.append(singleConnection)

for i in range(100,150):
    singleConnection = ((i, 0, 1, delay))
    loopConnections.append(singleConnection)
    singleConnection = ((i, 1, 20, delay))
    loopConnections.append(singleConnection)

for i in range(150,160):
    singleConnection = ((i, 0, 1, delay))
    loopConnections.append(singleConnection)
    singleConnection = ((i, 1, 10, delay))
    loopConnections.append(singleConnection)


for i in range(160,170):
    singleConnection = ((i, 0, 10, delay))
    loopConnections.append(singleConnection)
    singleConnection = ((i, 1, 1, delay))
    loopConnections.append(singleConnection)

for i in range(170,220):
    singleConnection = ((i, 0, 20, delay))
    loopConnections.append(singleConnection)
    singleConnection = ((i, 1, 1, delay))
    loopConnections.append(singleConnection)

for i in range(220, 320):
    singleConnection = ((i, 0, 60, delay))
    loopConnections.append(singleConnection)
    singleConnection = ((i, 1, 1, delay))
    loopConnections.append(singleConnection)


populations.append(
    p.Population(2, p.IF_curr_exp(**cell_params_lif), label='pop_1'))
populations.append(
    p.Population(2, p.IF_curr_exp(**cell_params_lif1), label='pop_2'))
populations.append(
    p.Population(2, p.external_devices.SpikeInjector(), label='inputSpikes_1'))

#projections.append(p.Projection(
#    populations[0], populations[1], p.FromListConnector(loopConnections),
#    ))
connections = [(0,0,40,0.01),(0,1,120,0.01),(1,0,2400,0.01),(1,1,40,0.01)]
projections.append(p.Projection(populations[0], populations[1], p.FromListConnector(connections),
    p.StaticSynapse(weight=20, delay=1.0)))
projections.append(p.Projection( populations[2], populations[0], p.OneToOneConnector(),
    p.StaticSynapse(weight=40, delay=1.0)))


populations[0].record(['spikes','v','gsyn_exc'])
populations[1].record(['spikes','v','gsyn_exc'])


p.external_devices.activate_live_output_for(populations[1])

live_connection = p.external_devices.SpynnakerLiveSpikesConnection(
    receive_labels=['pop_2'], send_labels=["inputSpikes_1"])
neuron0 = 0
neuron1 = 0
lastspiketime = 0
current_milli_time = lambda: int(round(time.time() * 1000))
leftspikes = 0
rightspikes = 0

def receive_spikes_1(label, time, neuron_ids):
   print("Received spikes from population {}, neurons {} at time {}".format(
        label, neuron_ids, time))
      
   #global neuron0,neuron1,neuron2,neuron3,neuron4,neuron5,lastspiketime
   global leftspikes, rightspikes
   #t = current_milli_time()
   #if t >= lastspiketime + 100:
   #     printRatesAndResetCounter()
   #lastspiketime = t
   if 0 in neuron_ids:
    leftspikes = leftspikes +1
   if 1 in neuron_ids:
    rightspikes = rightspikes + 1

def printRatesAndResetCounter():
   global neuron0,neuron1,neuron2,neuron3,neuron4,neuron5,lastspiketime
   neuron0rate = neuron0/float(2)
   neuron1rate = neuron1/float(2)
   print "neuron0 rate = ", neuron0rate
   print "neuron1 rate = ", neuron1rate
   neuron0 = 0
   neuron1 = 0
          
newlist = list()

def send_spikes(label, sender):
    #time.sleep(0.01)
    global newlist
    print("Sending spike to neurons: ", newlist)
    sender.send_spikes(label, newlist)



while True:
 print("\nInput neurons to spike\n")
 s = sys.stdin.readline()
 if "end" in s:
  break
 newlist = [int(elem) for elem in s.split()]
 init = time.time()
 live_connection.add_receive_callback('pop_2', receive_spikes_1)
 live_connection.add_start_callback("inputSpikes_1", send_spikes)
 p.run(runtime)
# print("leftspikes = ",leftspikes, "rightspikes = ", rightspikes)
 curr = time.time()
 exectime = curr-init
 print "execution time ", exectime
 print "leftspikes:",leftspikes," rightspikes:",rightspikes
 leftspikes = 0
 rightspikes = 0
 
 

#print projections[0].get(["weight", "delay"], format="list")

#print projections[1].get(["weight", "delay"], format="list")


#spikes_1 = populations[0].get_data('spikes')
#v = populations[0].get_data('v')
data = populations[0].get_data(['v', 'gsyn_exc', 'spikes'])
#spikes_2 = populations[1].get_data('spikes')
#v2 = populations[1].get_data('v')
data1 = populations[1].get_data(['v', 'gsyn_exc', 'spikes'])
print(type(data1.segments[0].filter(name='v')[0][0]))
print(len(data1.segments[0].filter(name='v')[0]))
#print((data1.segments[0].filter(name='v')[0][0]))

line_properties = [{'color': 'red', 'markersize': 2},
                   {'color': 'blue', 'markersize': 2}]


arr = data1.segments[0].filter(name='v')[0]
arr1 = data1.segments[0].filter(name='gsyn_exc')[0]

fig = plt.figure(1)
plt.plot(arr.times, arr[:, 0] )

plt.figure(2)
plt.plot(arr.times, arr[:, 1] )

plt.figure(3)
plt.plot(arr1.times, arr1[:, 0] )

plt.figure(4)
plt.plot(arr1.times, arr1[:, 1] )

plt.show()

Figure(
   # raster plot of the presynaptic neuron spike times
   Panel(data.segments[0].spiketrains,yticks=True,xticks=True, line_properties=line_properties, xlim=(0, runtime)),
   Panel(data.segments[0].filter(name='v')[0],ylabel="Membrane potential (mV)",yticks=True, xlim=(0, runtime)),
  # Panel(data.segments[0].filter(name='gsyn_exc')[0],ylabel="current",yticks=True, xlim=(0, runtime)),
   Panel(data1.segments[0].spiketrains,yticks=True,xticks=True, line_properties=line_properties, xlim=(0, runtime)),
#   Panel(v1,ylabel="Membrane potential (mV)",yticks=True, xlim=(0, runtime)),
   Panel(data1.segments[0].filter(name='gsyn_exc')[0],ylabel="current",yticks=True, xlim=(0, runtime)),
#   title="DNF",
   annotations=connections
)


plt.show()
p.end()
