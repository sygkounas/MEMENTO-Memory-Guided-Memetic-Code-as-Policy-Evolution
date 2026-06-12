from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import json
import signal

from utils import *
from inference import evaluate_policy, make_env
import time

client = get_client()


def load_previous_pbest(out_dir: Path):
    gens = sorted(out_dir.glob("gen_*"))
    if not gens:
        return None, None, None, None, None

    last = gens[-1]

    policy_path = last / "pbest_policy.txt"
    eval_path = last / "pbest_eval.json"

    if not policy_path.exists() or not eval_path.exists():
        return None, None, None, None, None

    code = policy_path.read_text()
    data = json.loads(eval_path.read_text())

    return (
        code,
        float(data["fitness"]),
        data["metrics"],
        data["success"],
        int(last.name.split("_")[1]),
    )


# ----------------------------
# Config
# ----------------------------
@dataclass
class Cfg:
    project_root: str = "/home/alkis/Downloads/robotsuite/cap_multi_evo/hanoi_4x4"
    prompts_dir: str = "prompts"
    out_dir: str = "runs/run_000"

    generations: int = 6          # total generations INCLUDING gen 0
    k_hill: int = 10
    k_macro: int = 10
    k_cross: int = 4

    n_episodes: int = 3
    horizon: int = 10000

    prompt_init: str = "policy_init.txt"
    prompt_hill: str = "hill_climb_prompt.txt"
    prompt_macro: str = "macro_mutation_prompt.txt"
    prompt_cross: str = "crossover_prompt.txt"


# ----------------------------
# Helpers
# ----------------------------
class StopFlag:
    def __init__(self):
        self.stop = False
        self._old = None

    def __enter__(self):
        self._old = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        return self

    def _handler(self, signum, frame):
        self.stop = True

    def __exit__(self, exc_type, exc, tb):
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)
        return False


def save_generation(gen_dir: Path, code: str, fitness: float, success, metrics) -> None:
    write_text(gen_dir / "pbest_policy.txt", code)
    write_json(
        gen_dir / "pbest_eval.json",
        {
            "fitness": float(fitness),
            "success": success,
            "metrics": metrics,
        },
    )


def run_one_generation(
    gen: int,
    cfg: Cfg,
    out: Path,
    hill_tmpl: str,
    macro_tmpl,
    cross_tmpl,

    parent_code: str,
    parent_fit: float,
    parent_met,
    parent_success,
    sf: StopFlag,
    env,
):
    gen_dir = out / f"gen_{gen:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = gen_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    # ----------------------------
    # Sequential hill climbing
    # ----------------------------
    cur_code = parent_code
    cur_fit = parent_fit
    cur_met = parent_met
    curr_success = parent_success
    failed_history = []

    for i in range(cfg.k_hill):
        if sf.stop:
            break

        hill_dir = temp_dir / f"hill_{i:02d}"
        hill_dir.mkdir(exist_ok=True)

        prompt = fill_template_hill(
            hill_tmpl,
            policy_code=cur_code,
            metrics=cur_met,
            fitness=cur_fit,
            failed_hill_climbing=failed_history,
        )


        write_text(hill_dir / "prompt.txt", prompt)

        raw = call_model(client, prompt)
        write_text(hill_dir / "raw_output.txt", raw)

        code = extract_policy_code(raw)
        write_text(hill_dir / "extracted_policy.py", code)

        try:
            assert_contains_policy(code)
            success, fit, met = evaluate_policy(
                code,
                env=env,
                n_episodes=cfg.n_episodes,
            )
        except Exception as e:
            write_text(hill_dir / "error.txt", str(e))
            continue

        if fit >= cur_fit:
            cur_code = code
            cur_fit = float(fit)
            cur_met = met
            curr_success = success
            print(f"[Gen {gen:03d} | Hill {i:02d}] Improved fitness: {cur_fit:.3f} (success={curr_success})")
        else:
            print(f"[Gen {gen:03d} | Hill {i:02d}] No improvement: {fit:.3f} vs current {cur_fit:.3f} (success={success} vs current {curr_success})")
            history_text = get_hill_history_from_code(code)
            failed_history.append(history_text)

    hill_code_best = cur_code
    hill_fit_best = cur_fit
    hill_met_best = cur_met
    hill_success_best = curr_success

    # ----------------------------
    # Macro mutation
    # ----------------------------
    macro_best_code = parent_code
    macro_best_fit = parent_fit
    macro_best_met = parent_met
    macro_best_success = parent_success

    if macro_tmpl is not None:
        for i in range(cfg.k_macro):
            if sf.stop:
                break

            macro_dir = temp_dir / f"macro_{i:02d}"
            macro_dir.mkdir(exist_ok=True)

            prompt = fill_template_macro(
                macro_tmpl,
                policy_code=parent_code,
                metrics=parent_met,
                fitness=parent_fit,
            )

            write_text(macro_dir / "prompt.txt", prompt)

            raw = call_model(client, prompt)
            write_text(macro_dir / "raw_output.txt", raw)

            code = extract_policy_code(raw)
            write_text(macro_dir / "extracted_policy.py", code)

            try:
                assert_contains_policy(code)
                success, fit, met = evaluate_policy(
                    code,
                    env=env,
                    n_episodes=cfg.n_episodes,
                )
            except Exception as e:
                write_text(macro_dir / "error.txt", str(e))
                continue

            if fit >= macro_best_fit:
                macro_best_code = code
                macro_best_fit = float(fit)
                macro_best_met = met
                macro_best_success = success
                print(f"[Gen {gen:03d} | Macro {i:02d}] New macro best fitness: {macro_best_fit:.3f} (success={macro_best_success})")
            else:
                print(f"[Gen {gen:03d} | Macro {i:02d}] No improvement over parent: {fit:.3f} vs parent {parent_fit:.3f} (success={success} vs parent {parent_success})")

    # ----------------------------
    # Crossover proposals
    # ----------------------------
    cross_best_fit = -1
    cross_best_code = None
    cross_best_met = None
    cross_best_success = None

    if cross_tmpl is not None:
        parent_a = hill_code_best
        parent_b = macro_best_code

        for i in range(cfg.k_cross):
            if sf.stop:
                break

            cross_dir = temp_dir / f"cross_{i:02d}"
            cross_dir.mkdir(exist_ok=True)

            prompt = fill_template_cross(
                cross_tmpl,
                policy_code1=parent_a,
                metrics1=hill_met_best,
                fitness1=hill_fit_best,
                policy_code2=parent_b,
                metrics2=macro_best_met,
                fitness2=macro_best_fit,
            )

            write_text(cross_dir / "prompt.txt", prompt)

            raw = call_model(client, prompt)
            write_text(cross_dir / "raw_output.txt", raw)

            code = extract_policy_code(raw)
            write_text(cross_dir / "extracted_policy.py", code)

            try:
                assert_contains_policy(code)
                success, fit, met = evaluate_policy(
                    code,
                    env=env,
                    n_episodes=cfg.n_episodes,
                )
            except Exception as e:
                
                
                write_text(cross_dir / "error.txt", str(e))
                continue

            if fit >= cross_best_fit:
                cross_best_fit = float(fit)
                cross_best_code = code
                cross_best_met = met
                cross_best_success = success
                print(f"[Gen {gen:03d} | Cross {i:02d}] New cross best fitness: {cross_best_fit:.3f} (success={cross_best_success})")

    # ----------------------------
    # Select best among hill / macro / cross
    # ----------------------------
    best_fit = hill_fit_best
    pbest_code = hill_code_best
    pbest_fitness = hill_fit_best
    pbest_metrics = hill_met_best
    pbest_success = hill_success_best

    if macro_best_fit >= best_fit:
        best_fit = macro_best_fit
        pbest_code = macro_best_code
        pbest_fitness = macro_best_fit
        pbest_metrics = macro_best_met
        pbest_success = macro_best_success
        print(f"[Gen {gen:03d}] Macro mutation is better than hill best {pbest_fitness:.3f} (success={pbest_success})")

    if cross_best_fit >= best_fit:
        best_fit = cross_best_fit
        pbest_code = cross_best_code
        pbest_fitness = cross_best_fit
        pbest_metrics = cross_best_met
        pbest_success = cross_best_success
        print(f"[Gen {gen:03d}] Crossover is better than current best {pbest_fitness:.3f} (success={pbest_success})")

    save_generation(gen_dir, pbest_code, pbest_fitness, pbest_success, pbest_metrics)

    return pbest_code, pbest_fitness, pbest_metrics, pbest_success


def initialize_generation_zero(
    cfg: Cfg,
    prompts_dir: Path,
    out: Path,
    hill_tmpl: str,
    macro_tmpl,
    cross_tmpl,
    sf: StopFlag,
    env,
):
    init_prompt_path = prompts_dir / cfg.prompt_init
    if not init_prompt_path.exists():
        raise RuntimeError(f"Missing init prompt: {init_prompt_path}")

    gen_dir = out / "gen_000"
    gen_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = gen_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    init_dir = temp_dir / "init"
    init_dir.mkdir(exist_ok=True)

    prompt = read_text(init_prompt_path)
    write_text(init_dir / "prompt.txt", prompt)

    raw = call_model(client, prompt)
    write_text(init_dir / "raw_output.txt", raw)

    code = extract_policy_code(raw)
    write_text(init_dir / "extracted_policy.py", code)

    try:
        assert_contains_policy(code)
        success, fit, met = evaluate_policy(
            code,
            env=env,
            n_episodes=cfg.n_episodes,
        )
    except Exception as e:
        write_text(init_dir / "error.txt", str(e))
        raise RuntimeError(f"Generation 0 initialization failed: {e}")

    write_text(init_dir / "parent_policy.txt", code)
    write_json(
        init_dir / "parent_eval.json",
        {
            "fitness": float(fit),
            "success": success,
            "metrics": met,
        },
    )

    print(f"[Gen 000 | Init] Parent fitness: {float(fit):.3f} (success={success})")

    return run_one_generation(
        gen=0,
        cfg=cfg,
        out=out,
        hill_tmpl=hill_tmpl,
        macro_tmpl=macro_tmpl,
        cross_tmpl=cross_tmpl,
        parent_code=code,
        parent_fit=float(fit),
        parent_met=met,
        parent_success=success,
        sf=sf,
        env=env,
    )


# ----------------------------
# Main
# ----------------------------
def main(cfg: Cfg) -> None:
    root = Path(cfg.project_root)
    prompts_dir = root / cfg.prompts_dir
    out = root / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    debug_root = out / "debug"
    debug_root.mkdir(exist_ok=True)

    hill_tmpl = read_text(prompts_dir / cfg.prompt_hill)

    macro_path = prompts_dir / cfg.prompt_macro
    cross_path = prompts_dir / cfg.prompt_cross

    macro_tmpl = None
    cross_tmpl = None

    if cross_path.exists():
        cross_tmpl = read_text(cross_path)
    if macro_path.exists():
        macro_tmpl = read_text(macro_path)

    env = make_env(render=False)
    try:
        with StopFlag() as sf:
            prev = load_previous_pbest(out)

            # ----------------------------
            # No generations yet -> run gen_000 from policy_init
            # ----------------------------
            if prev[0] is None:
                print("No generations found. Initializing gen_000 from policy_init.txt")
                pbest_code, pbest_fitness, pbest_metrics, pbest_success = initialize_generation_zero(
                    cfg=cfg,
                    prompts_dir=prompts_dir,
                    out=out,
                    hill_tmpl=hill_tmpl,
                    macro_tmpl=macro_tmpl,
                    cross_tmpl=cross_tmpl,
                    sf=sf,
                    env=env,
                )
                start_gen = 1
            else:
                pbest_code, pbest_fitness, pbest_metrics, pbest_success, last_gen_index = prev
                print(f"Resuming from gen_{last_gen_index:03d} with fitness {pbest_fitness} and success {pbest_success}")
                start_gen = last_gen_index + 1

            # ----------------------------
            # Continue with normal generations
            # ----------------------------
            for gen in range(start_gen, cfg.generations):
                if sf.stop:
                    break

                pbest_code, pbest_fitness, pbest_metrics, pbest_success = run_one_generation(
                    gen=gen,
                    cfg=cfg,
                    out=out,
                    hill_tmpl=hill_tmpl,
                    macro_tmpl=macro_tmpl,
                    cross_tmpl=cross_tmpl,
                    parent_code=pbest_code,
                    parent_fit=pbest_fitness,
                    parent_met=pbest_metrics,
                    parent_success=pbest_success,
                    sf=sf,
                    env=env,
                )
    finally:
        env.close()


if __name__ == "__main__":
    main(Cfg())