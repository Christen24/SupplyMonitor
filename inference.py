"""Baseline inference script for the SCRM OpenEnv environment.

Compliant with the Meta PyTorch Hackathon pre-submission checklist.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from server.environment import SupplyChainEnvironment
from models import SCRMAction as Action, ActionType
from tasks.scenarios import TaskGrader
from utils.helpers import clamp_open_unit_interval, safe_score

#  Checklist Environment Variables 

# Set defaults ONLY for API_BASE_URL and MODEL_NAME
API_BASE_URL = os.getenv("API_BASE_URL", "https://christen24-supply-chain-risk-management.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

# HF_TOKEN must NOT have a default string
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - for docker image usage
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

#  Heuristic fallback 

def heuristic_action(obs_dict: dict, step: int, task_id: str) -> Action:
    """Simple rule-based agent used when no API key is available."""
    risk_events = obs_dict.get("risk_events", [])
    otd = obs_dict.get("delivery_performance", {}).get("on_time_delivery", 1.0)

    if risk_events:
        active_events = [e for e in risk_events if e.get("status") == "active"]
        if active_events:
            evt = active_events[0]
            if step % 2 == 0:
                return Action(action_type=ActionType.MITIGATE, action_data={"event_id": evt["id"]})
            else:
                return Action(action_type=ActionType.RECOVER, action_data={"event_id": evt["id"]})

    if otd < 0.90:
        return Action(action_type=ActionType.DIVERSIFY)
    if step % 4 == 0:
        suppliers = obs_dict.get("supplier_status", {})
        if suppliers:
            return Action(action_type=ActionType.ASSESS_RISK, supplier_id=next(iter(suppliers)))

    return Action(action_type=ActionType.MONITOR)

#  LLM Agent 

SYSTEM_PROMPT = """You are an expert supply chain risk manager. You receive an
observation of the current supply chain state and must choose the best action.

Available actions (return EXACTLY one JSON object):
- {"action_type": "monitor"}
- {"action_type": "assess_risk", "supplier_id": "<id>"}
- {"action_type": "mitigate", "action_data": {"event_id": "<id>"}}
- {"action_type": "diversify", "action_data": {"region": "<region>"}}
- {"action_type": "negotiate", "supplier_id": "<id>"}
- {"action_type": "recover", "action_data": {"event_id": "<id>"}}
- {"action_type": "update_sop"}
- {"action_type": "flag_for_exec"}

Respond with ONLY a valid JSON object, no explanation."""

def llm_action(client: OpenAI, obs_dict: dict) -> Action | None:
    """Ask the LLM to pick an action using the OpenAI client."""
    obs_summary = json.dumps(obs_dict, indent=2, default=str)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current supply chain state:\n{obs_summary}\n\nChoose your action:"},
            ],
            max_tokens=200,
            temperature=0.01,
        )
        raw = response.choices[0].message.content or "{}"
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        return Action.model_validate(json.loads(raw))
    except Exception as e:
        if "429" in str(e):
            print(f"  [RATE LIMIT] Falling back to heuristic.")
            return None
        print(f"  [LLM ERROR] {e}")
        return Action(action_type=ActionType.MONITOR)

#  Runner 

def run_task(task_id: str, client: OpenAI | None) -> float:
    """Run a single task with structured logs."""
    env = SupplyChainEnvironment()
    max_steps_dict = {"easy": 20, "medium": 35, "hard": 50}
    
    # [START] Log
    print(f"[START] env=supply-chain-risk-management task={task_id} model={MODEL_NAME}")
    
    obs = env.reset(task_config={"id": task_id, "max_steps": max_steps_dict[task_id]})
    obs_dict = obs.model_dump()
    grader = TaskGrader()
    trajectory = []
    
    step = 0
    while True:
        step += 1
        
        # Determine if we use LLM
        active_risks = any(e.get("status") == "active" for e in obs_dict.get("risk_events", []))
        otd = obs_dict.get("delivery_performance", {}).get("on_time_delivery", 1.0)
        llm_trigger = active_risks or (otd < 0.90) or (step % 5 == 0)

        action = None
        if client and llm_trigger:
            action = llm_action(client, obs_dict)
        
        if action is None:
            action = heuristic_action(obs_dict, step, task_id)

        # [STEP] Log - Required format
        print(f"[STEP] step={step} action={action.action_type.value}")

        obs = env.step(action)
        reward = obs.reward or 0.0
        done = obs.done
        obs_dict = obs.model_dump()

        trajectory.append({
            "observation": obs_dict,
            "action": action.model_dump(),
            "reward": reward,
            "reward_breakdown": obs_dict.get("reward_breakdown")
        })

        if done:
            break

    raw_v = grader.grade(task_id, trajectory, env.state.model_dump())
    score = safe_score(raw_v)
    
    # [END] Log
    rewards_list = [round(float(t.get("reward", 0.0)), 4) for t in trajectory]
    success_val = score >= 0.5
    print(f"[END] env=supply-chain-risk-management task={task_id} model={MODEL_NAME} success={success_val} steps={step} score={score:.4f} rewards={rewards_list}")
    return score

def main() -> None:
    # Initialize OpenAI client as required by checklist
    client = None
    if HF_TOKEN and OpenAI is not None:
        client = OpenAI(
            base_url=f"{API_BASE_URL}/v1" if not API_BASE_URL.endswith("/v1") else API_BASE_URL,
            api_key=HF_TOKEN,
        )
        print(f"Using LLM: {MODEL_NAME}")
    elif HF_TOKEN:
        print("HF_TOKEN is set, but the openai package is not installed; using heuristic agent")
    else:
        print("Using heuristic agent (HF_TOKEN not set)")

    scores: dict[str, float] = {}
    tasks = ["easy", "medium", "hard"]

    for task_id in tasks:
        scores[task_id] = run_task(task_id, client)

    print("\nFINAL SCORES:")
    for task_id in tasks:
        # Final safety net: ensure score is strictly in (0, 1)
        s = scores[task_id]
        if s <= 0.0:
            s = 0.05
        if s >= 1.0:
            s = 0.95
        scores[task_id] = s
        print(f"  {task_id}: {s:.4f}")

if __name__ == "__main__":
    main()
