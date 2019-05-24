import paramiko
import os,sys,time
import pexpect
import re

child = pexpect.spawn('ssh spiky@134.59.130.214')
print type(child)
child.logfile = open("/tmp/mylog", "w")
child.expect('spiky\@spiky-server:\~\$')
child.sendline('spinn4_PyNN08')
child.expect('\(spinn4_PyNN08\).*spiky\@spiky-server:\~\$')
child.sendline('cd ajames')


child1 = pexpect.spawn('ssh spiky@134.59.130.214')
child1.logfile = open("/tmp/mylog1", "w")
child1.expect('spiky\@spiky-server:\~\$')
child1.sendline('spinn4_PyNN08')
child1.expect('\(spinn4_PyNN08\).*spiky\@spiky-server:\~\$')
child1.sendline('cd ajames')


child.sendline('python sender.py')
child.expect('WARNING:')
time.sleep(0.1)
child1.sendline('python DNF.py')
time.sleep(2)
child.sendline("\r\n")
child.expect('Input neurons to spike')
child.sendline('1 2 3 4')
child1.sendline("\r\n")
while True:
 try:
     child1.expect('\n')
     #print child1.before
     line = child1.before
     #print(line)
     x=re.match("leftfreq = ([\.\d]+).*rightfreq = ([\.\d]+)",line)
     if x:
         print x.group(1), x.group(2)
 except pexpect.exceptions.TIMEOUT:
     break

        
child.sendline('exit')
child1.sendline('exit')
