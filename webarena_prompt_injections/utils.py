# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import os
import re
import copy
from typing import List


def slug(text: str, max_len: int = 60) -> str:
    """Lowercase, trim, and reduce arbitrary text to a filename/id-safe token.

    Used to build the human-readable, globally-unique ``record_id`` that keys the
    per-template results (a bare ``task_id`` restarts at 1000 in every grid cell and
    would collapse when rows are aggregated across cells).
    """
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:max_len] or "unnamed"


def read_json(path_to_read_from: str):
    with open(path_to_read_from, "r") as json_file:
        return json.load(json_file)


def load_prompt_injection_config(config_file_path: str) -> List[dict]:
    """
    Reads and parses the prompt injection config JSON file.
    Args:
        config_file_path (str): The path to the prompt injection config JSON file.
    Returns:
        List[dict]: A list of dictionaries representing the prompt injections.
    """
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)

    return config_data


def write_json(dict_object_to_write, path_to_write_to: str):
    with open(path_to_write_to, "w") as json_file:
        json.dump(dict_object_to_write, json_file, indent=4)


def write_json_with_task_ids_as_individual_files(
    list_of_dict_objects_to_write, path_to_write_to: str
):
    if type(list_of_dict_objects_to_write) != list:
        raise ValueError("This function is meant to write a list as individual files.")

    for dict_object in list_of_dict_objects_to_write:
        index = dict_object["task_id"]
        full_path_to_write_to = os.path.join(path_to_write_to, f"{index}.json")
        with open(full_path_to_write_to, "w") as json_file:
            json.dump(dict_object, json_file, indent=4)


# Output formats whose agent LLM is NOT controlled by the WASP --model flag: these
# run the PTE ReAct/CodeAct scaffolding, which sources its model from PTE's
# config/config.yaml (agent_llm_provider + agent_llm_model). Recording the --model
# flag for these mislabels the run (e.g. a claude-sonnet-4-6 agent stamped "gpt-4o").
_PTE_BACKED_FORMATS = frozenset({"react_api_web", "pte", "beyond_browsing", "claude_code"})


def _default_pte_dir() -> str:
    # PTE is a sibling of the wasp/ repo root, two levels up from this file
    # (mirrors prompt_injector._prep_pte_agent_script).
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "PTE"))


def read_pte_agent_model(pte_dir: str = None):
    """Return the agent model PTE actually uses, as 'provider/model' (e.g.
    'anthropic/claude-sonnet-4-6'), read from {pte_dir}/config/config.yaml.

    Parsed line-by-line (skipping '#'-commented lines) rather than with PyYAML so
    this works from the WASP venv, which does not depend on yaml. Returns None if the
    file is missing/unreadable or has no uncommented agent_llm_model.
    """
    pte_dir = pte_dir or _default_pte_dir()
    cfg_path = os.path.join(pte_dir, "config", "config.yaml")
    provider = model = None
    try:
        with open(cfg_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                m = re.match(r"agent_llm_provider:\s*(\S+)", stripped)
                if m and provider is None:
                    provider = m.group(1)
                m = re.match(r"agent_llm_model:\s*(\S+)", stripped)
                if m and model is None:
                    model = m.group(1)
    except Exception:
        return None
    if not model:
        return None
    return f"{provider}/{model}" if provider else model


def resolve_effective_model(output_format: str, requested_model: str, pte_dir: str = None):
    """The model that actually drives the agent, for truthful provenance.

    For PTE-backed formats the WASP --model flag is ignored by the runner, so return
    the real model from PTE config/config.yaml (falling back to the flag if it can't
    be read). For every other format the flag IS authoritative — return it unchanged.
    """
    if output_format not in _PTE_BACKED_FORMATS:
        return requested_model
    return read_pte_agent_model(pte_dir) or requested_model


def get_absolute_path_to_sibling_directory_with_name(sibling_dir_name: str):
    cwd = os.getcwd()
    sibling_directory = os.path.join(cwd, "..", sibling_dir_name)
    return os.path.abspath(sibling_directory)


def write_bash_script(path_to_script: str, content_of_script: str):
    with open(path_to_script, "w") as file:
        file.write(content_of_script)

    os.chmod(path_to_script, 0o755)  # rwxr-xr-x permissions


def mkdir_in_output_folder_and_return_absolute_path(output_dir: str, new_sub_dir: str):
    absolute_path_to_new_subdir = os.path.abspath(os.path.join(output_dir, new_sub_dir))
    os.makedirs(absolute_path_to_new_subdir, exist_ok=True)
    return absolute_path_to_new_subdir


def instantiate_dict_str_with_params(webarena_task_field: dict, params_dict: dict):

    def dict_dfs(D, params):
        if isinstance(D, (list, dict)):
            items = enumerate(D) if isinstance(D, list) else D.items()
            for k, v in items:
                if isinstance(v, str):
                    D[k] = v.format(**params)
                elif isinstance(v, (dict, list)):
                    dict_dfs(v, params)

    instantiated_webarena_task_field = copy.deepcopy(webarena_task_field)
    dict_dfs(instantiated_webarena_task_field, params_dict)

    return instantiated_webarena_task_field