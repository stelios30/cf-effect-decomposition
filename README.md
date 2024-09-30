# Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making

This repository contains the code and instructions necessary to replicate the results from the paper "*Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making*".

## Reproducibility

In this section, we show how to reproduce the results for each of the reported environments.

### Sepsis Experiment

To reproduce the results for the sepsis experiments, you will need to do the following:

- **Generate the MDP parameters for AI and clinician (CL) actors**. To do so, run the `notebooks/sepsis/learn_sepsis_mdp.ipynb` notebook, which will save the results under `results/sepsis/mdp_ai.pkl` and `results/sepsis/mdp_original.pkl`.
- **Learn AI and CL policies**. To do so, run the `notebooks/sepsis/learn_sepsis_actors.ipynb` notebook, which will train and save the policies under `results/sepsis/ai_policy.pkl` and `results/sepsis/cl_policy.pkl`.
- **Generate the Sepsis results** by running the following command:
  ```bash
  python -m ced.scripts.sepsis_experiment 8854 \
    --artifacts-dir results/sepsis \
    --mdp-path results/sepsis/mdp_original.pkl \
    --cl-policy-path results/sepsis/cl_policy.pkl \
    --ai-policy-path results/sepsis/ai_policy.pkl \
    --tcfe-threshold 0.8 \
    --num-trajectories 100 \
    --num-cf-samples 100 \
    --num-cf-samples-cond 20 \
    --max-horizon 40 \
    --trust-values 0.0,0.2,0.4,0.6,0.8,1.0 \
    --round-difference 5 \
    --reverse-sse-threshold 0.1\
    --reverse-sse-variance-threshold 0.01
  ```
- **Visualize the results** by running the `notebooks/sepsis/sepsis_results.ipynb` notebook.

The **time** needed to learn both MDP parameters is around $5$ hours whereas the time needed to run the the sepsis experiment is about $4$ hours.

### Gridworld Experiment

To reproduce the results for the Gridworld experiment, you will need to do the following:

- **Setup the LLM Planner**. To make the procedure easier we have provided a [docker-compose](https://docs.docker.com/compose/) configuration. You can find the necessary information in the `llm_setup.ipynb` notebook.
- **Learn Agent's policies**. First, make sure to have [GuildAI](https://guild.ai/) installed on your system, which we use for our experiment tracking. Next, to learn the policy for actor 1, it suffices to run `guild run ifa:train mode=a1` (where `ifa` stands for an instruction-following agent), copying the resulting policy to `results/grid/a1_policy.pt`. Likewise, to learn the policy for actor 1, simply set `mode=a2`, copying the resulting policy to `results/grid/a2_policy.pt`. The hyperparameters of both policies were obtained through the hyperparameter optimization procedure, which you can run by executing `guild run ifa:tune --optimizer random --max-trials 50`, where we selected the combination with the highest cumulative test reward averaged across all instructions.
- **Generate the Gridworld results**. We report our results across 5 random seeds: 5656992596, 7989549204, 4429586919, 9986573471, 4459386742. For full reproducibility, please run the experiment for the seed 5656992596 first, which will in turn sample and save the trajectories in `results/grid/trajectories.pkl` as well as calculate quantities for different interventions we consider under `results/grid/p4_trajectories.csv` (intervening on the A2's action) and `results/grid/p3_trajectories.csv` (intervening on the Planner's action). The remainder of the seeds can then be run in any order. To obtain results for a particular seed, it suffices to run the following command:
    ```bash
    python -m ced.scripts.grid_experiment run \
        --results-dir results/grid \
        --env-file .env \
        --num-trajectories 10 \
        --num-cf-samples 100 \
        --num-cf-samples-cond 20 \
        --horizon 25 \
        --device cpu \
        --seed {SEED}
    ```
- **Visualize the results** by running the `notebooks/grid/grid_results.ipynb` notebook.

The **time** needed to learn A1 and A2 policies is approximately ~4h on a single NVIDIA A40 GPU. The runtime of the main experiment is severely bottlenecked by the inference capacity of the LLM planner. In our experiments on a single GPU, the sampling of a single trajectory took approximately ~5.5s, out of which ~5s were spent on LLM's inference. Hence, obtaining results for a single seed took approximately 2.5 days.

## Dependencies

This project depends on Python version 3.9.13. To install the necessary dependencies it suffices to run `pip install -r requirements.txt`.
