# Experiments on the GPU cluster using HTCondor

This section describes the use of condor on the cluster and provides some scripts to make it easier.

# Introduction

Using condor provides a faster alternative then using jupyter-notebooks.
We are able to run each experiment in parallel thanks to condor.
By that we can increase the number of running experiments by the number of available GPUs.
Also letting the jobs run in the background gives us time for other tasks.


***Requirements for the following sections are that a conda environment exists.***


# Creating a condor job

Its good practice to keep a separate folder for each condor job.
A condor job consists of at least 2 files.
We need a *.sub file to describe the requirements and also specify the files to run.
The second file is the one we want to run in our case a *.py file.

A basic condor job can look like this:
```
executable = /bin/bash 
arguments = -i conda.sh minape main.py
log = logfile.log 
transfer_input_files = conda.sh, main.py
output = stdout.txt 
error = stderr.txt 
should_transfer_files = IF_NEEDED 
request_gpus = 1 
queue
```


We need to use a shell script to launch our job with a conda environment.
It looks like this:
```
conda activate $1 
python ${@:2}
```

We can specify the conda environment inside the *.sub or *.sh file.

So the file structure now looks like this:
 ```
base/
│
├── main.sub
├── main.py
├── conda.sh
```

*You can find this example in the base folder in the main condor experiments.*

__For detailed Information on the *.sub file please read the documentation.__


# Submitting a condor job

We created our folder structure in the last step.
Now all we need to do is submit our job.

[Connect to the cluster first.](# Connecting to the cluster)

Now we run the command __from inside the folder__.
```
condor_submit main.sub
````
> Submitting job(s).\
> 1 job(s) submitted to cluster 777.


We can watch our job now with the command:

```
condor_q
```

To get live updates we can use the command:

```
watch -n 1 condor_q
```
The Job can have multiple states, the most important are IDLE, RUN, HOLD.

IDLE: Waiting for free node.

RUN: The job is running on the cluster.

HOLD: There was some error: check log file and stderr.

Another good command to know is,
```
condor_rm 777
```
which removes the specified job.
We need to do this if our job is on HOLD.


# Connecting to the cluster

To connect to the condor node on the cluster do the following:

1. Connect to the gateway (gate01 or gate02):
```
ssh username@gate01.l3s.uni-hannover.de
```
2. Connect to the desk node (desk01 or desk02):
```
ssh username@pascal-desk01.l3s.uni-hannover.de
```
3. Finally connect to the condor node (condor01 or condor02):
```
ssh username@pascal-condor01.l3s.intra
```

We can check that we are on the correct node by running any condor command for example:
```
condor_q
```

# Included scripts

This repo contains some scripts to make our life easier.

For the main experiments the main script is run.sh located inside the main folder.

The script makes it easier to run multiple experiments without going into each folder and running them separately.

Remember to give permission to the script first by running:
```
chmod +x ./run.sh
```

We need to this once.

Then we can use the script like this:
```
./run.sh -f 7 15 33
```
This will run the experiments 7, 15 and 33 on the cluster.

The script takes folder names as input, only for the argument '-f'.

---
Next are the scripts inside the influence_revcolconvmixer folder:
 ```
influence_revcolconvmixer/
│
├── run_scripts/
│   ├── run_26_cx.sh
│   ├── run_26_dx.sh
│   ├── run_30_cx.sh
│   ├── run_30_dx.sh
│   ├── run_36_cx.sh
│   └── run_36_dx.sh
├── 26_c2
├── ...
├── ...
├── 36_d6
└── run.sh
```
The "run.sh" script is the same as before.

The ones inside the "run_scripts" folder are custom scripts that run each version specified in the filename.

For example "run_26_cx.sh" will run: 26_c2, 26_c3, 26_c4, 26_c5, 26_c6.

You need to run all scripts from inside their respective folder.


# Batch queue

Let's say we want to run the same file multiple times like for the influence experiments.

For that all we need to do is to give a value after the queue line.

For example 10 runs would be:

```
queue 10
```

We also need to make sure that running them in parallel does not lead to conflicts with file access.


# Example: running a main experiment

You shouldn't need to change anything besides path directories inside the experiment file you want to run (almost always dataset).

Example 1:
```
ssh username@gate01.l3s.uni-hannover.de
ssh username@pascal-desk01.l3s.uni-hannover.de
ssh username@pascal-condor01.l3s.intra

cd ../MinApe/experiments/condor/main/
./run.sh -f 7
watch -n 1 condor_q
```

Example 2:
```
ssh username@gate01.l3s.uni-hannover.de
ssh username@pascal-desk01.l3s.uni-hannover.de
ssh username@pascal-condor01.l3s.intra

cd ../MinApe/experiments/condor/main/
cd 7/
condor_submit main.sub
condor_q
watch -n 1 condor_q
cat stdout.txt
```

Output will be inside the specified experiment folder in the "stdout.txt" file, after the job is completed.

