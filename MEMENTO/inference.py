import numpy as np

number = 0

# Load environment
with open(".../env_tower_heavy_dr.py", "r") as f:
    exec(f.read(), globals())  # defines FrankaEnv


def make_env(render=False):
    kwargs = dict(
        robots="Panda",
        use_camera_obs=False,
        use_object_obs=True,
        has_renderer=render,
        has_offscreen_renderer=False,
        hard_reset=False,
        ignore_done=False,
    )
    return FrankaEnv(**kwargs)


def evaluate_policy(policy_code, env, n_episodes=3):
    local_ns = {}
    exec(policy_code, local_ns)
    policy = local_ns["FrankaPolicy"]()

    episode_fitnesses = []
    all_metrics = []
    successes = 0

    for ep in range(n_episodes):
        obs = env.reset()
        policy.reset()

        for t in range(env.horizon):
            action = policy.compute_action(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break

        episode_fitness, metrics = env.get_episode_fitness()
        episode_fitnesses.append(episode_fitness)
        all_metrics.append(metrics)

        success = int(metrics["success"])
        successes += success
        print(f"Episode {ep+1}/{n_episodes} - Success: {success} - Fitness: {episode_fitness:.4f} - Metrics: {metrics}")

    return successes / n_episodes, float(np.mean(episode_fitnesses)), all_metrics



# from pathlib import Path
# import json

# if __name__ == "__main__":
#     project_root = Path("/home/alkis/Downloads/robotsuite/cap_multi_evo/hanoi_4x4")
#     #policy_path = project_root / "runs" / "run_000_seed3" / "gen_003" / "pbest_policy.txt"
#     policy_path = project_root / "runs" / "run_000_no_cross_working" / "gen_005" / "pbest_policy.txt"
    
    
# #/home/alkis/Downloads/robotsuite/cap_multi_evo/hanoi_4x4/runs/run_000_no_cross_working/gen_005
# #/home/alkis/Downloads/robotsuite/cap_multi_evo/hanoi_4x4/runs/run_000_no_cross_working/gen_005

#     policy_code = policy_path.read_text(encoding="utf-8")

#     env = make_env(render=True)

#     success_rate, fitness, metrics = evaluate_policy(
#         policy_code=policy_code,
#         env=env,
#         n_episodes=6,
#     )

#     out_dir = project_root / "runs" / "run_000" / "gen_002"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     (out_dir / "pbest_policy.txt").write_text(policy_code, encoding="utf-8")

#     (out_dir / "pbest_eval.json").write_text(
#         json.dumps(
#             {
#                 "fitness": fitness,
#                 "success": success_rate,
#                 "metrics": metrics,
#             },
#             indent=2,
#         ),
#         encoding="utf-8",
#     )

#     env.close()