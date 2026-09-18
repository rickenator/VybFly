Energy curves across scales - precision note (read this before quoting a joule figure)

TWO RUNS OF THE SAME PROTOCOL

  scale   run A (J)   run B (J)   delta     J/spike (B)   capability/watt (B)
    1x      469.39      437.60     -6.8%       8.89 uJ         0.0703
    2x      516.38      500.26     -3.1%       5.20 uJ         0.6152
    5x      984.31      945.71     -3.9%       3.92 uJ         0.6122
   10x     2120.27     2021.63     -4.7%       4.19 uJ         0.5259

Run A is archived as results/energy/curves_stale_v2capability.json (its capability block was
also stale - see below). Run B is the delivered results/energy/curves.json.

WHAT IS AND IS NOT REPRODUCIBLE

* Spike and synaptic-event counts reproduce to within 0.05% (e.g. 49,198,504 spikes at 1x in
  both runs), so the workload itself is deterministic.
* GPU joules do NOT: they vary by 3-7% between runs because the number is an integral of sampled
  instantaneous power (5 Hz NVML) on a shared, thermally-variable device, not a counter. Quote
  joules with ~5% uncertainty, or quote the spike/event counts, which are exact.
* Consequence for the heading finding: the fall in cost per spike from 1x to 10x is ~2.1-2.2x in
  both runs (9.54 -> 4.40 uJ in A, 8.89 -> 4.19 uJ in B), which is far larger than the run-to-run
  spread, so that conclusion is robust. The absolute values are not.

CORRECTION TO AN EARLIER CLAIM

An earlier report in this project stated the energy numbers "reproduced exactly to three
decimals" between runs. That was wrong: the two readings being compared were both taken from run
A's output file, not from two independent runs. The genuine repeat is run B above, and it differs
by up to 6.8%. The claim is corrected here and the numbers above are what should be cited.

CAPABILITY BLOCK PROVENANCE

Run A's capability values were copied from phase 9's superseded protocol-v2 file (memory
capacity 8/96/96/96 with the older metric set). Run B reads the current
results/phase9/capability.json, whose protocol fingerprint is 39f6c75cde6548e8. That phase 9
flag was correct and is fixed.

CENSORING

memory_capacity is 8 at 1x and 96 at 2x/5x/10x, and 96 is the ceiling of the tested class grid,
so every capability-per-watt figure from 2x upward is a LOWER BOUND on capability and therefore a
lower bound on capability per watt. Extending the class grid is the prerequisite for any honest
capability-scaling exponent on that metric.
