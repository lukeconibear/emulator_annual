#!/bin/bash
# submit looped chain of jobs
# run as . custom_outputs_regrid_batch.bash
# each is <10 minutes job for 2,500 custom outputs

current=$(qsub regrid_to_popgrid.bash)
echo $current

for id in {2..100}; do
  current_id=$(echo $current | tr -d -c 0-9)
  next=$(qsub -hold_jid $current_id regrid_to_popgrid.bash)
  echo $next
  current=$next;
done
