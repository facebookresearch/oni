# oni

This repository contains a Pytorch implementation of [**Online Intrinsic Rewards for Decision Making Agents from Large Language Model Feedback**](https://arxiv.org/abs/2410.23022)
by [Qinqing Zheng*](https://enosair.github.io/), [Mikael Henaff*](https://www.mikaelhenaff.com/), [Amy Zhang](https://amyzhang.github.io/), [Aditya Grover](https://aditya-grover.github.io/) and [Brandon Amos](https://bamos.github.io/).

If you use this code for your research, please cite us as:
```Bibtex
@article{zheng2024online,
  title={Online Intrinsic Rewards for Decision Making Agents from Large Language Model Feedback},
  author={Zheng, Qinqing and Henaff, Mikael and Zhang, Amy and Grover, Aditya and Amos, Brandon},
  journal={arXiv preprint arXiv:2410.23022},
  year={2024}
}
```

## Overview
ONI is a distributed architecture (*builds on top of [sample-factory](https://github.com/alex-petrenko/sample-factory)*) that simultaneously learns an RL policy and an intrinsic reward function using LLM feedback. Our approach annotates the agent's collected experience via an asynchronous LLM server,  which is then distilled into an intrinsic reward model. 

![system](https://github.com/user-attachments/assets/d78a580a-d7a3-4cb6-b41c-21abebfa375d)

We support a range of algorithmic choices for reward modeling with varying complexity, including hashing, classification, and ranking models. Our approach achieves state-of-the-art performance across a range of challenging, sparse reward tasks from the NetHack Learning Environment in a simple unified process, solely using the agent's gathered experience, without requiring external datasets.

![main_results_return](https://github.com/user-attachments/assets/b7e17490-e33b-4b45-87df-b76cedf08779)





## Envrionment Installation
```console
conda create -f env.yml
```
## Usage
Suppose you have a node with 2 GPUs. We first launch the LLM server on `GPU-0` using Llama-3.1-8B:
```console
export NUM_GPU=1; export MODEL_DIR=YOUR_MODEL_DIR; export MODEL=Meta-Llama-3.1-8B-Instruct; ./scripts/launch_fastchat_server.sh $MODEL_DIR/$MODEL $NUM_GPU &> fastchat.log
```
Then, we can train an RL agent for the `Score` task on `GPU-1`, using `ONI-Retrieval`:
```console
export CUDA_VISIBLE_DEVICES=1;python scripts/main.py \                                                               
    --train_dir YOUR_DIR_TO_SAVE_TRAINING_PROGRESS_AND_LOG \
    --llm_model Meta-Llama-3.1-8B-Instruct \
    --llm_server_addr localhost \
    --experiment default \
    --root_env NetHackScoreExtendedActions-v1 \
    --llm_reward_type online_cls \
    --llm_reward_coeff 0.4 \
    --extrinsic_reward_coeff 0.1 \
    --wandb True \
    --wandb_entity YOUR_WANDB_ENTITY \
    --wandb_proj YOUR_WANDB_PROJ
```
* The arguments `llm_server_addr` and `llm_model` specify the HTTP address and model type of the LLM server. Our code uses (FastChat)[https://github.com/lm-sys/FastChat] so it supports cross-node communication, see the example below.

* The `root_env` argument specifies the nethack envrionment (task). We use `NetHackScoreExtendedActions-v1`, `NetHackOracleExtendedActions-v1`, `NetHackStaircaseLvl3ExtendedActions-v1` and `NetHackStaircaseLvl4ExtendedActions-v1` for our paper. 

* The `llm_reward_coeff` and `extrinsic_reward_coeff` are coefficients multipled to intrinsic rewards and environment-provided extrnsic rewards. In our paper, we set `extrinsic_reward_coeff` to 0.1 for `NetHackScoreExtendedActions-v1` and 10 for the others.

* The *reward free* setting in our paper uses the `NetHackScoreExtendedActions-v1` environment with `extrinsic_reward_coeff=0`.

* `llm_reward_type` controls the type of intrinsic rewards. Currently we support
  - `online_cls` ==> ONI-Retrieval
  - `online_cls_reward_model` ==> ONI-Classification
  - `online_motif` ==> ONI-Ranking
  - `motif` ==> offline motif, see [here](https://github.com/facebookresearch/motif)
  - `cosine-bow` ==> ELLM-BoW
  - `none` ==> extrinsic reward only

  Please check out our paper to see the descriptions of those methods.

### Cross-Node Communication
If we host the LLM server on another node, all we need to change is the `llm_server_addr`. Below we provide an example sbatch script to set this up using [heterogenous jobs](https://slurm.schedmd.com/heterogeneous_jobs.html).
```console
#!/bin/bash
#SBATCH --output=./slurm.out
#SBATCH --job-name=example_job
#SBATCH --time=48:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=YOUR_PARTITION
#SBATCH --cpus-per-task=50

#SBATCH hetjob

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=YOUR_PARTITION
#SBATCH --cpus-per-task=50

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
     echo "bypass sigterm"
   else
     echo "Requeuing " $SLURM_JOB_ID
     scontrol requeue $SLURM_JOB_ID
   fi
}

trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

export LOGDIR=YOUR_LOGDIR

mkdir -p $LOGDIR

echo $LOGDIR

MODEL=Meta-Llama-3.1-8B-Instruct
MODEL_DIR=YOUR_MODEL_DIR
NUM_GPU=1

srun --het-group=0 \
     --output ${LOGDIR}/server_%j.out \
     ./scripts/launch_fastchat_server.sh ${MODEL_DIR}/${MODEL} $NUM_GPU &
SERVER_PID=$!

echo "Server is starting at $SLURM_JOB_NODELIST_HET_GROUP_0:9001, waiting to start the client"
while ! curl -s $SLURM_JOB_NODELIST_HET_GROUP_0:9001/v1/models | grep -q $MODEL; do
    echo "... still offline"
    sleep 10s
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "Server died, exiting"
        exit 1
    fi
done
echo "Server online, starting client job"

srun --het-group=1 \
    --output ${LOGDIR}/appo_%j.out \
    python ./scripts/main.py \
            --train_dir $LOGDIR \
            --llm_model $MODEL \
            --llm_server_addr $SLURM_JOB_NODELIST_HET_GROUP_0 \
            $@
```

## License
The majority of `oni` is licensed under CC-BY-NC, however portions of the project are
available under separate license terms:
* sample-factory - MIT License

## Acknowledgements
This repository builds heavily off of [sample-factory](https://github.com/alex-petrenko/sample-factory), [motif](https://github.com/facebookresearch/motif)
