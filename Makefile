.PHONY: inference-demo clean

inference-demo:
	cd inference-demo && sbatch run.sub

clean:
	cd inference-demo && rm slurm-out/* posterior_medians.png samples.csv
