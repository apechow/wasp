# Copyright (c) Meta Platforms, Inc. and affiliates.
import click
import subprocess
import json
from collections import defaultdict
from constants import (
    PromptInjectionFormat,
    GitlabUserGoals,
    RedditUserGoals
)


def run_single_end_to_end(
        config,  # prompt injection configurations
        model,  # agentic model to run
        system_prompt,  # system prompt for agent
        output_dir,  # to store settings, logs, errors
        output_format,  # agentic scaffolding format
        user_goal_idx,  # benign user goal
        injection_format,  # prompt injection format to try
        results_dict,  # to save and accumulate results
        output_dir_idx=0,  # to save logs for each run separately
        multi_docker=False,  # use multi-docker worker pool
    ):
    if output_dir[-1] == '/':
        output_dir = output_dir + str(output_dir_idx) + '/'
    else:
        output_dir = output_dir + '/' + str(output_dir_idx) + '/'

    command = [
        'bash',
        'scripts/run_end_to_end.sh',
        output_dir,
        model,
        system_prompt,
        config,
        str(user_goal_idx),
        injection_format,
        output_format,
        '--multi-docker' if multi_docker else '',
    ]
    print(f"\nRunning command: \n{' '.join([str(arg) for arg in command])}", flush=True)
    
    # Run the main command
    subprocess.run(command, check=True)

    # ------- check and save results
    try:
        with open("/tmp/run_step_by_step_asr.json", 'r') as json_file:
            res_step_by_step = json.load(json_file)
            if "compromised" in res_step_by_step:
                results_dict["cnt_intermediate_compromised"] += res_step_by_step["compromised"]
            if "not_compromised" in res_step_by_step:
                results_dict["cnt_intermediate_not_compromised"] += res_step_by_step["not_compromised"]
            if "outright_refusal" in res_step_by_step:
                results_dict["cnt_intermediate_outright_refusal"] += res_step_by_step["outright_refusal"]
    except FileNotFoundError:
        print("WARNING: /tmp/run_step_by_step_asr.json not found — step-by-step ASR metrics skipped")

    try:
        with open("/tmp/run_attacker_utility.json", 'r') as json_file:
            res_attacker_utility = json.load(json_file)
            results_dict["cnt_end2end_compromised"] += res_attacker_utility["total_scores"]
            results_dict["cnt_end2end_not_compromised"] += (res_attacker_utility["cnt_tasks"] - res_attacker_utility["total_scores"])
    except FileNotFoundError:
        print("WARNING: /tmp/run_attacker_utility.json not found — attacker utility metrics skipped")

    try:
        with open("/tmp/run_user_utility.json", 'r') as json_file:
            res_user_utility = json.load(json_file)
            results_dict["cnt_user_utility"] += res_user_utility["total_scores"]
            results_dict["cnt_user_total_tasks"] += res_user_utility["cnt_tasks"]
    except FileNotFoundError:
        print("WARNING: /tmp/run_user_utility.json not found — user utility metrics skipped")
    # -------


def run_all(config,
            model,
            system_prompt,
            output_dir,
            output_format,
            run_single,
            user_goal_start,
            multi_docker=False):
    gitlab_user_goals = GitlabUserGoals("")
    reddit_user_goals = RedditUserGoals("")
    assert len(gitlab_user_goals.GOALS) == len(reddit_user_goals.GOALS), "Number of user goals should match!"
    user_goals_len = len(gitlab_user_goals.GOALS)
    injection_format_list = [PromptInjectionFormat.GOAL_HIJACKING_PLAIN_TEXT,
                             PromptInjectionFormat.GOAL_HIJACKING_URL_INJECTION]
    results_dict = defaultdict(int)

    for user_goal_idx in range(user_goal_start, user_goals_len):
        print(f"$$$$$$$ Running {user_goal_idx+1} our of {user_goals_len} user goals, current one: "
              f"(gitlab) '{gitlab_user_goals.GOALS[user_goal_idx]}', "
              f"(reddit) '{reddit_user_goals.GOALS[user_goal_idx]}'")
        for i, injection_format in enumerate(injection_format_list):
            print(f"$$$$$$$ Running {i+1} out of {len(injection_format_list)} injection formats, current one: {injection_format}")

            run_single_end_to_end(config=config,
                                  model=model,
                                  system_prompt=system_prompt,
                                  output_dir=output_dir,
                                  output_format=output_format,
                                  user_goal_idx=user_goal_idx,
                                  injection_format=injection_format,
                                  results_dict=results_dict,
                                  output_dir_idx=user_goal_idx * len(injection_format_list) + i,
                                  multi_docker=multi_docker)

            print(f"\nAccumulated results after user_goal_idx = {user_goal_idx+1} and injection_format_idx = {i+1}: ")
            for key, value in results_dict.items():
                print(f"{key} = {value}")

            if run_single:
                print("\n!!! Running a single user goal and a single injection format is requested. Terminating")
                return
    
    print("\n\nDone running all experiments! Final results:")
    for key, value in results_dict.items():
        print(f"{key} = {value}")


@click.command()
@click.option(
    "--config",
    type=str,
    default="configs/experiment_config.raw.json",
    help="Where to find the config for prompt injections",
)
@click.option(
    "--model",
    type=click.Choice(['gpt-4o', 'gpt-4o-mini', 'claude-35', 'claude-37'], case_sensitive=False),
    default="gpt-4o",
    help="backbone LLM. Available options: gpt-4o, gpt-4o-mini, claude-35, claude-37",
)
@click.option(
    "--system-prompt",
    type=str,
    default="configs/system_prompts/wa_p_som_cot_id_actree_3s.json",
    help="system_prompt for the backbone LLM. Default = VWA's SOM system prompt for GPT scaffolding",
)
@click.option(
    "--output-dir",
    type=str,
    default="/tmp/computer-use-agent-logs",
    help="Folder to store the output configs and commands to run the agent",
)
@click.option(
    "--output-format",
    type=str,
    default="webarena",
    help="Format of the agentic scaffolding: webarena (default), claude, gpt_web_tools, pte, beyond_browsing",
)
@click.option(
    "--run-single",
    is_flag=True,
    default=False,
    help="whether to test only a single user goal and a single injection format",
)
@click.option(
    "--user_goal_start",
    type=int,
    default=0,
    help="starting user_goal index (between 0 and total number of benign user goals)",
)
@click.option(
    "--multi-docker",
    is_flag=True,
    default=False,
    help="Use the multi-docker worker pool for parallel task execution (pte and beyond_browsing formats only)",
)
def main(config,
         model,
         system_prompt,
         output_dir,
         output_format,
         run_single,
         user_goal_start,
         multi_docker):
    print("Arguments provided to run.py: \n", locals(), "\n\n")
    run_all(config=config,
            model=model,
            system_prompt=system_prompt,
            output_dir=output_dir,
            output_format=output_format,
            run_single=run_single,
            user_goal_start=user_goal_start,
            multi_docker=multi_docker)


if __name__ == '__main__':
    main()
