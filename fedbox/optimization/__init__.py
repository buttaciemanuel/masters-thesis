'''
This submodule defines the three optimization algorithms that can
be used in the simulations. The structure of the code is the same
for each algorithm, and we always have two main components.
<ol>
  <li>A <b>Coordinator</b> instance which represents the central server 
  broadcasting the model at the beginning of every round and performing the aggregation of
  the received updates before the beginning of the next.</li>
  <li>A <b>Agent</b> instance which performs local optimization and evaluation on its data samples 
  with a specific federated framework.</li>
</ol>
For each algorithmic framework, we extend from the base classes <b>Coordinator</b> and <b>Agent</b> and
we specialize them accordingly to the specific procedure.
'''