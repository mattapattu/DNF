
import pyNN.spiNNaker as p
import time
import sys
import logging

logging.basicConfig()


def send_spikes(label, sender):
 while True:
  print("\nInput neurons to spike\n")
  s = sys.stdin.readline()
  s = s.splitlines()
  s = [i for i in s if i]
  if s:
   s = s[0]
  else :
   continue
  print "Taking first line only",s
  if "end" in s:
   break
  newlist = [int(elem) for elem in s.split()]
  sender.send_spikes(label,newlist)

live_connection = p.external_devices.SpynnakerLiveSpikesConnection(
    send_labels=["inputSpikes_1"], local_port=19990)
live_connection.add_start_callback("inputSpikes_1", send_spikes)

live_connection.join()
