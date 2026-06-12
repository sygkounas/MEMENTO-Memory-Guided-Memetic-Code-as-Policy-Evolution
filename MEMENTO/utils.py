from pathlib import Path
import json
import numpy as np
from openai import OpenAI
from typing import Any, Dict, Tuple, List, Optional
import re
MODEL = "gpt-5.2"

def append_human_feedback(prompt: str, feedback: str) -> str:
    if not feedback.strip():
        return prompt
    return (
        prompt
        + "\n\n----- HUMAN FEEDBACK -----\n"
        + feedback.strip()
        + "\n"
    )



def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def fill_template_cross(
    tmpl: str,
    *,
    policy_code1: str, metrics1: Any, fitness1: Any,
    policy_code2: str, metrics2: Any, fitness2: Any,
) -> str:
    def _m(x: Any) -> str:
        if isinstance(x, (dict, list)):
            return json.dumps(x, indent=2, sort_keys=True)
        return str(x)

    return (
        tmpl.replace("{policy_code1}", policy_code1)
            .replace("{Metrics1}", _m(metrics1))
            .replace("{Fitness1}", str(fitness1))
            .replace("{policy_code2}", policy_code2)
            .replace("{Metrics2}", _m(metrics2))
            .replace("{Fitness2}", str(fitness2))
    )
def get_hill_history_from_code(code: str) -> str:
    try:
        local_ns = {}
        exec(code, {"np": __import__("numpy")}, local_ns)
        PolicyCls = local_ns.get("FrankaPolicy")
        if PolicyCls is None:
            return ""
        # avoid __init__ side effects / required args
        policy = object.__new__(PolicyCls)
        fn = getattr(policy, "hill_climb_history", None)
        if callable(fn):
            return str(fn())
        return ""
    except Exception:
        return ""




def fill_template_macro(tmpl: str, *, policy_code: str, metrics: Any, fitness: Any) -> str:
    if isinstance(metrics, (dict, list)):
        metrics_str = json.dumps(metrics, indent=2, sort_keys=True)
    else:
        metrics_str = str(metrics)
    return (
        tmpl.replace("{policy_code}", policy_code)
            .replace("{Metrics}", metrics_str)
            .replace("{Fitness}", str(fitness))
    )
    
    
def fill_template_hill(
    tmpl: str,
    *,
    policy_code: str,
    metrics: Any,
    fitness: Any,
    failed_hill_climbing: Optional[List[str]] = None,
) -> str:

    if isinstance(metrics, (dict, list)):
        metrics_str = json.dumps(metrics, indent=2, sort_keys=True)
    else:
        metrics_str = str(metrics)

    failed_str = ""
    if failed_hill_climbing:
        failed_str = "\n".join(failed_hill_climbing)

    return (
        tmpl.replace("{policy_code}", policy_code)
            .replace("{Metrics}", metrics_str)
            .replace("{Fitness}", str(fitness))
            .replace("{failed_hill_climbing}", failed_str)
    )


def extract_policy_code(raw: str) -> str:
    blocks = re.findall(r"```python\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    blocks = re.findall(r"```\s*(.*?)```", raw, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return raw.strip()

def assert_contains_policy(code: str) -> None:
    if "class FrankaPolicy" not in code:
        raise RuntimeError("LLM output does not contain 'class FrankaPolicy'")
# ----------------------------
# LLM
# ----------------------------
def get_client():
    return OpenAI()


def call_model(client, prompt: str):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="high",
    )
    return resp.choices[0].message.content


