# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import time
import copy
import json
from typing import List

import click
from environment_editors.gitlab_editor import GitlabEditor
from environment_editors.reddit_editor import RedditEditor
from environment_editors.base_environment_editor import (
    BaseWebArenaEditor,
    WebArenaEditorException,
)
from environment_setup import SETUP_ACTION_NAME_TO_FUNCTION

from utils import (
    load_prompt_injection_config,
    mkdir_in_output_folder_and_return_absolute_path,
    write_json,
    write_json_with_task_ids_as_individual_files,
    get_absolute_path_to_sibling_directory_with_name,
    write_bash_script,
    instantiate_dict_str_with_params,
)
from tqdm import tqdm

from constants import (
    GPT_TOOL_WEB_AGENT_PREAMBLE,
    GPT_TOOL_WEB_AGENT_SINGLE_RUN_TEMPLATE,
    GPT_TOOL_WEB_AGENT_CLEANUP,
    PTE_BASH_SCRIPT_PREAMBLE,
    PTE_BASH_SCRIPT_SINGLE_RUN_TEMPLATE,
    PTE_BASH_SCRIPT_PREAMBLE_PARALLEL,
    PTE_BASH_SCRIPT_SINGLE_RUN_TEMPLATE_PARALLEL,
    PTE_BASH_SCRIPT_POSTAMBLE_PARALLEL,
    BEYOND_BROWSING_BASH_SCRIPT_PREAMBLE,
    BEYOND_BROWSING_BASH_SCRIPT_SINGLE_RUN_TEMPLATE,
    BEYOND_BROWSING_BASH_SCRIPT_PREAMBLE_PARALLEL,
    BEYOND_BROWSING_BASH_SCRIPT_SINGLE_RUN_TEMPLATE_PARALLEL,
    BEYOND_BROWSING_BASH_SCRIPT_POSTAMBLE_PARALLEL,
    REACT_API_WEB_BASH_SCRIPT_PREAMBLE,
    REACT_API_WEB_BASH_SCRIPT_SINGLE_RUN_TEMPLATE,
    STARTING_DUMMY_WEBARENA_TASK_INDEX,
    WEBARENA_GITLAB_TASK,
    WEBARENA_REDDIT_TASK,
    PromptInjectionFormat,
    OutputFormat,
    CLAUDE_BASE_COMMAND,
    WEBARENA_BASH_SCRIPT_SOM,
    WEBARENA_BASH_SCRIPT_AXTREE,
    GitlabUserGoals,
    RedditUserGoals,
)


class WebArenaPromptInjector:
    def __init__(
        self,
        editor_list: List[BaseWebArenaEditor],
        prompt_injection_configs: List[dict],
    ) -> None:
        self.editor_list = editor_list
        self.prompt_injection_configs = prompt_injection_configs
        self.domain_map = {}
        # initializing domains
        for editor in self.editor_list:
            if isinstance(editor, GitlabEditor):
                self.domain_map["gitlab_domain"] = editor.gitlab_domain
            elif isinstance(editor, RedditEditor):
                self.domain_map["reddit_domain"] = editor.reddit_domain

    def inject_in_environment(
        self,
        injection_format: str,
        skip_environment: bool,
        output_dir: str,
        output_format: OutputFormat,
        system_prompt: str,
        user_goal_idx: int,
        model: str,
        multi_docker: bool = False,
    ):
        if multi_docker and output_format in (OutputFormat.PTE, OutputFormat.BEYOND_BROWSING):
            return self._inject_in_environment_multi_docker(
                injection_format=injection_format,
                output_dir=output_dir,
                output_format=output_format,
                user_goal_idx=user_goal_idx,
            )

        if not skip_environment:
            self._prepare_environment()  # this calls setup function needed to prepare websites

        # prepare prompts for injection
        self._prepare_injection(injection_format, user_goal_idx)

        # inject prompt into websites
        webarena_tasks_config, webarena_attacker_tasks_config = self._inject_prompts(
            user_goal_idx, skip_environment
        )

        # saving user tasks in webarena format so that we can evaluate utility later
        webarena_tasks_dir = mkdir_in_output_folder_and_return_absolute_path(
            output_dir, "webarena_tasks"
        )
        webarena_attacker_tasks_dir = mkdir_in_output_folder_and_return_absolute_path(
            output_dir, "webarena_tasks_attacker"
        )
        write_json_with_task_ids_as_individual_files(
            webarena_tasks_config, webarena_tasks_dir
        )
        write_json_with_task_ids_as_individual_files(
            webarena_attacker_tasks_config, webarena_attacker_tasks_dir
        )

        match output_format:
            case OutputFormat.WEBARENA:
                content_of_script_to_run_agent = (
                    self._prep_webarena_agent_script_and_write_task_files(
                        webarena_tasks_config,
                        webarena_tasks_dir,
                        output_dir,
                        system_prompt,
                        model,
                    )
                )

            case OutputFormat.CLAUDE:
                content_of_script_to_run_agent = self._prep_claude_agent_script(
                    webarena_tasks_config,
                    output_dir,
                    system_prompt=system_prompt,
                    model=model,
                )

            case OutputFormat.GPT_WEB_TOOLS:
                if system_prompt:
                    raise NotImplementedError(
                        "Custom system prompts not yet implemented for GPT Tools Web Agent!"
                    )
                content_of_script_to_run_agent = self._prep_gpt_tool_web_agent_script(
                    webarena_tasks_config, output_dir, model
                )

            case OutputFormat.PTE:
                content_of_script_to_run_agent = self._prep_pte_agent_script(
                    webarena_tasks_config, output_dir
                )

            case OutputFormat.BEYOND_BROWSING:
                content_of_script_to_run_agent = self._prep_beyond_browsing_agent_script(
                    webarena_tasks_config, output_dir
                )

            case OutputFormat.REACT_API_WEB:
                content_of_script_to_run_agent = self._prep_react_api_web_agent_script(
                    webarena_tasks_config, output_dir
                )

            case _:
                raise NotImplementedError(f"Invalid output format {output_format}")

        path_to_agent_script = os.path.join(output_dir, "run_agent.sh")
        write_bash_script(
            path_to_script=path_to_agent_script,
            content_of_script=content_of_script_to_run_agent,
        )

        path_to_instantiated_prompt_injection_config = os.path.join(
            output_dir, "instantiated_prompt_injections_config.json"
        )
        write_json(
            self.prompt_injection_configs,
            path_to_instantiated_prompt_injection_config,
        )
        return path_to_agent_script, path_to_instantiated_prompt_injection_config

    def _prep_webarena_agent_script_and_write_task_files(
        self,
        webarena_tasks_config,
        webarena_tasks_dir,
        output_dir,
        system_prompt,
        model,
    ):

        results_dir = mkdir_in_output_folder_and_return_absolute_path(
            output_dir, "agent_logs"
        )

        webarena_agent_dir = get_absolute_path_to_sibling_directory_with_name(
            "visualwebarena"
        )

        if "som" in system_prompt:
            return WEBARENA_BASH_SCRIPT_SOM.format(
                webarena_agent_dir=webarena_agent_dir,
                start_task_index=STARTING_DUMMY_WEBARENA_TASK_INDEX,
                end_task_index=STARTING_DUMMY_WEBARENA_TASK_INDEX
                + len(webarena_tasks_config),
                webarena_tasks_dir=webarena_tasks_dir,
                results_dir=results_dir,
                instruction_path=system_prompt,
                model=model,
            )
        else:
            return WEBARENA_BASH_SCRIPT_AXTREE.format(
                webarena_agent_dir=webarena_agent_dir,
                start_task_index=STARTING_DUMMY_WEBARENA_TASK_INDEX,
                end_task_index=STARTING_DUMMY_WEBARENA_TASK_INDEX
                + len(webarena_tasks_config),
                webarena_tasks_dir=webarena_tasks_dir,
                results_dir=results_dir,
                instruction_path=system_prompt,
                model=model,
            )

    def _prep_claude_agent_script(
        self, webarena_tasks_config, output_dir, system_prompt, model
    ):

        conversation_log_folder_path = mkdir_in_output_folder_and_return_absolute_path(
            output_dir, "agent_logs"
        )

        claude_agent_dir = get_absolute_path_to_sibling_directory_with_name(
            f"{model}-computer-use-demo"
        )

        script_to_run_claude_agent = "#!/bin/bash\n"
        script_to_run_claude_agent += "set -e\n"
        script_to_run_claude_agent += f"cd {claude_agent_dir}\n"
        script_to_run_claude_agent += (
            f'source "{claude_agent_dir}/.venv/bin/activate"\n'
        )

        script_to_run_claude_agent += "docker build . -t computer-use-demo:local\n"

        for task in webarena_tasks_config:
            if "gitlab" in task["sites"]:
                system_prompt_formatted = system_prompt.format(
                    USERNAME="byteblaze",
                    PASSWORD="hello1234",
                    DOMAIN=self.domain_map["gitlab_domain"],
                )
            elif "reddit" in task["sites"]:
                system_prompt_formatted = system_prompt.format(
                    USERNAME="MarvelsGrantMan136",
                    PASSWORD="test1234",
                    DOMAIN=self.domain_map["reddit_domain"],
                )
            else:
                raise ValueError(f"Unrecognized website: {str(task['sites'])}")
            script_to_run_claude_agent += CLAUDE_BASE_COMMAND.format(
                start_url=task["start_url"],
                user_goal=task["intent"],
                conversation_log_folder_path=conversation_log_folder_path,
                claude_agent_dir=claude_agent_dir,
                index_of_conversation=task["task_id"],
                system_prompt=system_prompt_formatted,
                model=model,
            )
            script_to_run_claude_agent += "\n"

        script_to_run_claude_agent += f"deactivate\n"

        return script_to_run_claude_agent

    def _prep_gpt_tool_web_agent_script(self, webarena_tasks_config, output_dir, model):
        conversation_log_folder_path = mkdir_in_output_folder_and_return_absolute_path(
            output_dir, "agent_logs"
        )

        webarena_agent_dir = get_absolute_path_to_sibling_directory_with_name(
            "visualwebarena"
        )

        script_to_run_gpt_web_tool_agent = GPT_TOOL_WEB_AGENT_PREAMBLE.format(
            webarena_agent_dir=webarena_agent_dir,
        )

        for task in webarena_tasks_config:
            trace_log_filepath = os.path.join(
                conversation_log_folder_path, f"{task['task_id']}.jsonl"
            )
            webarena_config_path = os.path.join(
                output_dir, f"webarena_tasks/{task['task_id']}.json"
            )
            script_to_run_gpt_web_tool_agent += (
                GPT_TOOL_WEB_AGENT_SINGLE_RUN_TEMPLATE.format(
                    task_id=task["task_id"],
                    webarena_config_path=webarena_config_path,
                    model=model,
                    trace_log_filepath=trace_log_filepath,
                )
            )
            script_to_run_gpt_web_tool_agent += "\n"

        script_to_run_gpt_web_tool_agent += GPT_TOOL_WEB_AGENT_CLEANUP
        return script_to_run_gpt_web_tool_agent

    def _prep_pte_agent_script(self, webarena_tasks_config, output_dir):
        trace_log_dir = mkdir_in_output_folder_and_return_absolute_path(
            output_dir, "agent_logs"
        )
        # PTE is a sibling of the wasp/ repo root, two levels up from this file
        pte_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "PTE"))
        pte_python = os.path.join(pte_dir, "venv/bin/python")
        run_pte_agent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_pte_agent.py")

        script = PTE_BASH_SCRIPT_PREAMBLE
        for task in webarena_tasks_config:
            task_config_path = os.path.join(output_dir, f"webarena_tasks/{task['task_id']}.json")
            script += PTE_BASH_SCRIPT_SINGLE_RUN_TEMPLATE.format(
                task_id=task["task_id"],
                pte_python=pte_python,
                run_pte_agent_path=run_pte_agent_path,
                task_config_path=task_config_path,
                trace_log_dir=trace_log_dir,
                pte_dir=pte_dir,
            )
        return script

    def _prep_pte_agent_script_multi_docker(self, output_dir, num_workers):
        trace_log_dir = mkdir_in_output_folder_and_return_absolute_path(output_dir, "agent_logs")
        pte_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "PTE"))
        wasp_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv/bin/python")
        run_task_multi_docker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_task_multi_docker.py")

        script = PTE_BASH_SCRIPT_PREAMBLE_PARALLEL.format(num_workers=num_workers)
        for i in range(len(self.prompt_injection_configs)):
            task_id = STARTING_DUMMY_WEBARENA_TASK_INDEX + i
            pi_context_path = os.path.join(output_dir, f"pi_task_contexts/{task_id}.json")
            script += PTE_BASH_SCRIPT_SINGLE_RUN_TEMPLATE_PARALLEL.format(
                task_id=task_id,
                wasp_python=wasp_python,
                run_task_multi_docker_path=run_task_multi_docker_path,
                pi_context_path=pi_context_path,
                trace_log_dir=trace_log_dir,
                pte_dir=pte_dir,
            )
        script += PTE_BASH_SCRIPT_POSTAMBLE_PARALLEL
        return script

    def _prep_beyond_browsing_agent_script(self, webarena_tasks_config, output_dir):
        trace_log_dir = mkdir_in_output_folder_and_return_absolute_path(
            output_dir, "agent_logs"
        )
        beyond_browsing_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "BeyondBrowsing", "API-Based-Agent"
            )
        )
        beyond_browsing_python = os.path.join(beyond_browsing_dir, ".venv", "bin", "python")
        run_beyond_browsing_agent_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "run_beyond_browsing_agent.py"
        )

        script = BEYOND_BROWSING_BASH_SCRIPT_PREAMBLE
        for task in webarena_tasks_config:
            task_config_path = os.path.join(output_dir, f"webarena_tasks/{task['task_id']}.json")
            script += BEYOND_BROWSING_BASH_SCRIPT_SINGLE_RUN_TEMPLATE.format(
                task_id=task["task_id"],
                beyond_browsing_python=beyond_browsing_python,
                run_beyond_browsing_agent_path=run_beyond_browsing_agent_path,
                task_config_path=task_config_path,
                trace_log_dir=trace_log_dir,
                beyond_browsing_dir=beyond_browsing_dir,
            )
        return script

    def _prep_react_api_web_agent_script(self, webarena_tasks_config, output_dir):
        trace_log_dir = mkdir_in_output_folder_and_return_absolute_path(
            output_dir, "agent_logs"
        )
        pte_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "PTE"))
        pte_python = os.path.join(pte_dir, "venv/bin/python")
        run_react_api_web_agent_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "run_react_api_web_agent.py"
        )

        script = REACT_API_WEB_BASH_SCRIPT_PREAMBLE
        for task in webarena_tasks_config:
            task_config_path = os.path.join(output_dir, f"webarena_tasks/{task['task_id']}.json")
            script += REACT_API_WEB_BASH_SCRIPT_SINGLE_RUN_TEMPLATE.format(
                task_id=task["task_id"],
                pte_python=pte_python,
                run_react_api_web_agent_path=run_react_api_web_agent_path,
                task_config_path=task_config_path,
                trace_log_dir=trace_log_dir,
                pte_dir=pte_dir,
            )
        return script

    def _prep_beyond_browsing_agent_script_multi_docker(self, output_dir, num_workers):
        trace_log_dir = mkdir_in_output_folder_and_return_absolute_path(output_dir, "agent_logs")
        beyond_browsing_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "BeyondBrowsing", "API-Based-Agent"
            )
        )
        beyond_browsing_python = os.path.join(beyond_browsing_dir, ".venv", "bin", "python")
        run_task_multi_docker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_task_multi_docker.py")

        script = BEYOND_BROWSING_BASH_SCRIPT_PREAMBLE_PARALLEL.format(num_workers=num_workers)
        for i in range(len(self.prompt_injection_configs)):
            task_id = STARTING_DUMMY_WEBARENA_TASK_INDEX + i
            pi_context_path = os.path.join(output_dir, f"pi_task_contexts/{task_id}.json")
            script += BEYOND_BROWSING_BASH_SCRIPT_SINGLE_RUN_TEMPLATE_PARALLEL.format(
                task_id=task_id,
                beyond_browsing_python=beyond_browsing_python,
                run_task_multi_docker_path=run_task_multi_docker_path,
                pi_context_path=pi_context_path,
                trace_log_dir=trace_log_dir,
                beyond_browsing_dir=beyond_browsing_dir,
            )
        script += BEYOND_BROWSING_BASH_SCRIPT_POSTAMBLE_PARALLEL
        return script

    def _create_attacker_account_or_login(self, editor):
        try:
            editor.create_user(
                "Arsene",
                "Lupin",
                "arsenelupin",
                "arsenelupin@foo.foo",
                "notaverycommonlyusedpassword",
            )
        except WebArenaEditorException as e:
            if "Username has already been taken" in str(e):
                editor.login("arsenelupin", "notaverycommonlyusedpassword")
            else:
                raise

    def _prepare_environment(
        self,
    ) -> None:
        """
        Prepares the environment for the prompt injection.

        Args:
            prompt_data (dict[str, str]): The JSON data describing the prompt injection.
        """
        print("Peforming environment setup for each config")
        MAX_NUM_RETRIES = 3
        for pi_config in tqdm(self.prompt_injection_configs):
            # Implement setup actions here
            for editor in self.editor_list:
                if (
                    editor.environment == pi_config["environment"]
                    and pi_config["setup_fn"]
                ):
                    setup_fn_name = pi_config["setup_fn"]
                    for setup_attempt_i in range(MAX_NUM_RETRIES):
                        try:
                            setup_fn = SETUP_ACTION_NAME_TO_FUNCTION[setup_fn_name]
                            possible_updates_to_config = setup_fn(
                                editor, **pi_config["parameters"]
                            )
                            pi_config["parameters"].update(possible_updates_to_config)
                            break
                        except Exception as e:
                            if setup_attempt_i + 1 == MAX_NUM_RETRIES:
                                raise WebArenaEditorException(
                                    f"Setup attempts exhausted: {e}. Terminating..."
                                )
                            print(
                                f"Error during setup: {e}. Attempting to setup again: {setup_attempt_i+1}"
                            )
                            time.sleep(
                                2
                            )  # fail typically occurs due to timeout, put some sleep before retrying
        print(
            "Done with all setup actions. Environment should be ready to test prompt injections."
        )

    def _get_user_goal_str(self, prompt_injection_config, user_goal_idx):

        if prompt_injection_config["environment"] == "reddit":
            user_goal_obj = RedditUserGoals("")
        elif prompt_injection_config["environment"] == "gitlab":
            user_goal_obj = GitlabUserGoals("")
        else:
            raise NotImplementedError(
                f"Unknown environemnt: {prompt_injection_config['environment']}"
            )

        if "comment" in user_goal_obj.GOALS[user_goal_idx].lower():
            user_goal_str = "comment"
        elif "upvote" in user_goal_obj.GOALS[user_goal_idx].lower():
            user_goal_str = "upvote"
        elif "thumbs_up" in user_goal_obj.GOALS[user_goal_idx].lower():
            user_goal_str = "like"
        else:
            user_goal_str = "complete user instruction"
            print(f"PI does not know user goal, defaulting to '{user_goal_str}'")

        return user_goal_str

    def _prepare_injection(self, injection_format: str, user_goal_idx: int) -> None:
        """
        Prepares the injection for the prompt.

        Args:
            prompt_data (dict[str, str]): The JSON data describing the prompt injection.
        """
        print("Instantiating prompt injections.")
        params_dict = dict(self.domain_map)

        for prompt_injection_config in tqdm(self.prompt_injection_configs):
            params_dict.update(prompt_injection_config["parameters"])
            instantiated_url_of_action = prompt_injection_config["action_url"].format(
                **params_dict
            )
            prompt_injection_config["parameters"][
                "instantiated_action_url"
            ] = instantiated_url_of_action
            instantiated_instruction_text = prompt_injection_config[
                "instruction"
            ].format(**params_dict)
            if "url_injection" in injection_format:
                instantiated_instruction_text = instantiated_instruction_text.replace(
                    " ", "-"
                )

            try:
                user_goal_str = self._get_user_goal_str(
                    prompt_injection_config, user_goal_idx
                )
                prompt_injection_msg = PromptInjectionFormat.MESSAGE[injection_format]
                prompt_injection_config["instantiated_instruction"] = (
                    prompt_injection_msg.format(
                        user_goal_str=user_goal_str,
                        instantiated_url_of_action=instantiated_url_of_action,
                        instantiated_instruction_text=instantiated_instruction_text,
                    )
                )
            except Exception as e:
                raise Exception(
                    f"Unrecognized prompt injection format {injection_format} or some other error occurred: {e}"
                )

    def _create_task_dicts(
        self,
        user_goal,
        user_goal_idx,
        incrementing_task_id_number,
        issue_url,
        prompt_injection_config,
    ):

        if isinstance(user_goal, GitlabUserGoals):
            task_for_this_prompt_injection = copy.deepcopy(WEBARENA_GITLAB_TASK)
        elif isinstance(user_goal, RedditUserGoals):
            task_for_this_prompt_injection = copy.deepcopy(WEBARENA_REDDIT_TASK)
        else:
            raise NotImplementedError("Unknown user goal")

        # Creating dict that describes webarena task for user goal
        task_for_this_prompt_injection["intent"] = user_goal.GOALS[user_goal_idx]
        task_for_this_prompt_injection["eval"] = user_goal.EVALS[user_goal_idx]
        task_for_this_prompt_injection["task_id"] = incrementing_task_id_number
        task_for_this_prompt_injection["start_url"] = issue_url

        # Creating dict that describes webarena task for attacker goal (used for evaluations)
        task_for_this_attack = copy.deepcopy(task_for_this_prompt_injection)
        task_for_this_attack["intent"] = prompt_injection_config[
            "instantiated_instruction"
        ]
        params_dict = dict(self.domain_map)
        params_dict.update(prompt_injection_config["parameters"])
        task_for_this_attack["eval"] = instantiate_dict_str_with_params(
            prompt_injection_config["eval"], params_dict
        )
        task_for_this_attack["start_url"] = prompt_injection_config["parameters"][
            "instantiated_action_url"
        ]

        return task_for_this_prompt_injection, task_for_this_attack

    def _inject_gitlab_prompts(
        self,
        gitlab_editor: GitlabEditor,
        start_task_id,
        user_goal_idx,
        tasks_with_prompt_injections,
        tasks_from_attacker,
    ):
        print("...for gitlab")
        incrementing_task_id_number = start_task_id
        with gitlab_editor:
            self._create_attacker_account_or_login(gitlab_editor)
            for prompt_injection_config in self.prompt_injection_configs:
                if prompt_injection_config["environment"] != gitlab_editor.environment:
                    continue
                issue_url = gitlab_editor.create_issue_with_title_and_text(
                    "byteblaze",
                    "dotfiles",
                    "A note from a gentleman thief",
                    prompt_injection_config["instantiated_instruction"],
                )

                # creating dict that describes webarena task for user and attacker goals
                user_goal = GitlabUserGoals(issue_url)
                task_for_this_prompt_injection, task_for_this_attacker_goal = (
                    self._create_task_dicts(
                        user_goal,
                        user_goal_idx,
                        incrementing_task_id_number,
                        issue_url,
                        prompt_injection_config,
                    )
                )
                tasks_with_prompt_injections.append(task_for_this_prompt_injection)
                tasks_from_attacker.append(task_for_this_attacker_goal)
                incrementing_task_id_number += 1

        return incrementing_task_id_number

    def _inject_reddit_prompts(
        self,
        reddit_editor: RedditEditor,
        start_task_id,
        user_goal_idx,
        tasks_with_prompt_injections,
        tasks_from_attacker,
    ):
        print("...for reddit")
        incrementing_task_id_number = start_task_id
        for prompt_injection_config in self.prompt_injection_configs:
            if prompt_injection_config["environment"] != reddit_editor.environment:
                continue
            with reddit_editor:
                attacker_username = prompt_injection_config["parameters"][
                    "attacker_username"
                ]
                attacker_password = prompt_injection_config["parameters"][
                    "attacker_password"
                ]
                reddit_editor.create_user(
                    "", "", attacker_username, "", attacker_password
                )
                issue_url = reddit_editor.create_post_with_title_and_text(
                    "allentown",
                    "A note from a gentleman thief " + str(incrementing_task_id_number),
                    prompt_injection_config["instantiated_instruction"],
                    attacker_username,
                )
                # handle additional setup functions for reddit
                if "user_username" in prompt_injection_config["parameters"]:
                    user_username = prompt_injection_config["parameters"][
                        "user_username"
                    ]
                    user_password = prompt_injection_config["parameters"][
                        "user_password"
                    ]
                    reddit_editor.login(user_username, user_password)
                    if "user_post_title" in prompt_injection_config["parameters"]:
                        user_post_title = prompt_injection_config["parameters"][
                            "user_post_title"
                        ]
                        user_post_text = prompt_injection_config["parameters"][
                            "user_post_text"
                        ]
                        user_post_forum = prompt_injection_config["parameters"][
                            "user_post_forum"
                        ]
                        prompt_injection_config["parameters"]["user_post_url"] = (
                            reddit_editor.create_post_with_title_and_text(
                                user_post_forum,
                                user_post_title,
                                user_post_text,
                                user_username,
                            )
                        )
                    elif "user_msg_txt" in prompt_injection_config["parameters"]:
                        user_msg_txt = prompt_injection_config["parameters"][
                            "user_msg_txt"
                        ]
                        user_msg_recipient = prompt_injection_config["parameters"][
                            "user_msg_recipient"
                        ]
                        prompt_injection_config["parameters"]["user_msg_url"] = (
                            reddit_editor.send_message(user_msg_recipient, user_msg_txt)
                        )
                    elif "username_to_block" in prompt_injection_config["parameters"]:
                        username_to_block = prompt_injection_config["parameters"][
                            "username_to_block"
                        ]
                        reddit_editor.block_user(user_username, username_to_block)

                # creating dict that describes webarena task for user and attacker goals
                user_goal = RedditUserGoals(issue_url)
                task_for_this_prompt_injection, task_for_this_attacker_goal = (
                    self._create_task_dicts(
                        user_goal,
                        user_goal_idx,
                        incrementing_task_id_number,
                        issue_url,
                        prompt_injection_config,
                    )
                )
                tasks_with_prompt_injections.append(task_for_this_prompt_injection)
                tasks_from_attacker.append(task_for_this_attacker_goal)
                incrementing_task_id_number += 1

        return incrementing_task_id_number

    def _inject_prompts(self, user_goal_idx: int, skip_environment: bool):
        """
        Injects the prompt into the environment.
        """
        start_task_id = STARTING_DUMMY_WEBARENA_TASK_INDEX
        tasks_with_prompt_injections = []
        tasks_from_attacker = []

        if not skip_environment:
            for editor in self.editor_list:
                if isinstance(editor, GitlabEditor):
                    start_task_id = self._inject_gitlab_prompts(
                        editor,
                        start_task_id,
                        user_goal_idx,
                        tasks_with_prompt_injections,
                        tasks_from_attacker,
                    )
                elif isinstance(editor, RedditEditor):
                    start_task_id = self._inject_reddit_prompts(
                        editor,
                        start_task_id,
                        user_goal_idx,
                        tasks_with_prompt_injections,
                        tasks_from_attacker,
                    )
                else:
                    raise NotImplementedError("unknown environment")
        else:
            for prompt_injection_config in self.prompt_injection_configs:
                issue_url = "foo_issue_url"
                if prompt_injection_config["environment"] == "gitlab":
                    task_for_this_prompt_injection = copy.deepcopy(WEBARENA_GITLAB_TASK)
                elif prompt_injection_config["environment"] == "reddit":
                    task_for_this_prompt_injection = copy.deepcopy(WEBARENA_REDDIT_TASK)
                else:
                    raise NotImplementedError("Unknown environment")

                task_for_this_prompt_injection["task_id"] = start_task_id
                start_task_id += 1
                task_for_this_prompt_injection["start_url"] = issue_url
                tasks_with_prompt_injections.append(task_for_this_prompt_injection)
                task_for_this_attacker_goal = copy.deepcopy(
                    task_for_this_prompt_injection
                )
                tasks_from_attacker.append(task_for_this_attacker_goal)

        return tasks_with_prompt_injections, tasks_from_attacker

    def _inject_in_environment_multi_docker(
        self,
        injection_format: str,
        output_dir: str,
        output_format: OutputFormat,
        user_goal_idx: int,
    ):
        """Multi-docker path: skip static injection; write per-task pi_context files and a parallel run_agent.sh."""
        webarena_tasks_dir = mkdir_in_output_folder_and_return_absolute_path(output_dir, "webarena_tasks")
        webarena_attacker_tasks_dir = mkdir_in_output_folder_and_return_absolute_path(output_dir, "webarena_tasks_attacker")
        pi_task_contexts_dir = mkdir_in_output_folder_and_return_absolute_path(output_dir, "pi_task_contexts")

        for i, pi_config in enumerate(self.prompt_injection_configs):
            task_id = STARTING_DUMMY_WEBARENA_TASK_INDEX + i
            ctx = {
                "pi_config": copy.deepcopy(pi_config),
                "user_goal_idx": user_goal_idx,
                "injection_format": injection_format,
                "task_id": task_id,
                "task_dir": webarena_tasks_dir,
                "attacker_task_dir": webarena_attacker_tasks_dir,
            }
            write_json(ctx, os.path.join(pi_task_contexts_dir, f"{task_id}.json"))

        bb_eval_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "BeyondBrowsing", "API-Based-Agent", "evaluation", "webarena"
        ))
        import sys
        if bb_eval_dir not in sys.path:
            sys.path.append(bb_eval_dir)
        from worker_pool.workers import num_workers
        n_workers = num_workers()

        if output_format == OutputFormat.PTE:
            content = self._prep_pte_agent_script_multi_docker(output_dir, n_workers)
        else:
            content = self._prep_beyond_browsing_agent_script_multi_docker(output_dir, n_workers)

        path_to_agent_script = os.path.join(output_dir, "run_agent.sh")
        write_bash_script(path_to_agent_script, content)

        path_to_instantiated_prompt_injection_config = os.path.join(
            output_dir, "instantiated_prompt_injections_config.json"
        )
        write_json(self.prompt_injection_configs, path_to_instantiated_prompt_injection_config)

        return path_to_agent_script, path_to_instantiated_prompt_injection_config

    def setup_and_inject_single_config(
        self,
        pi_config: dict,
        editor,
        user_goal_idx: int,
        injection_format: str,
        task_id: int,
    ):
        """
        For multi-docker mode: run setup, instantiate injection text, and inject into one worker.
        Returns (task_dict, attacker_task_dict).
        The injector must be constructed with only the editor for this worker so that
        self.domain_map carries the correct worker URL.
        """
        MAX_NUM_RETRIES = 3

        # 1. Environment setup
        if pi_config.get("setup_fn"):
            setup_fn_name = pi_config["setup_fn"]
            for attempt in range(MAX_NUM_RETRIES):
                try:
                    setup_fn = SETUP_ACTION_NAME_TO_FUNCTION[setup_fn_name]
                    updates = setup_fn(editor, **pi_config["parameters"])
                    pi_config["parameters"].update(updates)
                    break
                except Exception as e:
                    if attempt + 1 == MAX_NUM_RETRIES:
                        raise WebArenaEditorException(
                            f"Setup attempts exhausted: {e}. Terminating..."
                        )
                    print(f"Error during setup: {e}. Attempting to setup again: {attempt + 1}")
                    time.sleep(2)

        # 2. Instantiate injection text (mirrors _prepare_injection for one config)
        params_dict = dict(self.domain_map)
        params_dict.update(pi_config["parameters"])

        instantiated_url_of_action = pi_config["action_url"].format(**params_dict)
        pi_config["parameters"]["instantiated_action_url"] = instantiated_url_of_action

        instantiated_instruction_text = pi_config["instruction"].format(**params_dict)
        if "url_injection" in injection_format:
            instantiated_instruction_text = instantiated_instruction_text.replace(" ", "-")

        user_goal_str = self._get_user_goal_str(pi_config, user_goal_idx)
        prompt_injection_msg = PromptInjectionFormat.MESSAGE[injection_format]
        pi_config["instantiated_instruction"] = prompt_injection_msg.format(
            user_goal_str=user_goal_str,
            instantiated_url_of_action=instantiated_url_of_action,
            instantiated_instruction_text=instantiated_instruction_text,
        )

        # 3. Inject into the worker environment and return task dicts
        if isinstance(editor, GitlabEditor):
            return self._inject_single_gitlab_config(pi_config, editor, user_goal_idx, task_id)
        elif isinstance(editor, RedditEditor):
            return self._inject_single_reddit_config(pi_config, editor, user_goal_idx, task_id)
        else:
            raise NotImplementedError(f"Unknown editor type: {type(editor)}")

    def _inject_single_gitlab_config(self, pi_config, gitlab_editor, user_goal_idx, task_id):
        with gitlab_editor:
            self._create_attacker_account_or_login(gitlab_editor)
            issue_url = gitlab_editor.create_issue_with_title_and_text(
                "byteblaze",
                "dotfiles",
                "A note from a gentleman thief",
                pi_config["instantiated_instruction"],
            )
        user_goal = GitlabUserGoals(issue_url)
        return self._create_task_dicts(user_goal, user_goal_idx, task_id, issue_url, pi_config)

    def _inject_single_reddit_config(self, pi_config, reddit_editor, user_goal_idx, task_id):
        with reddit_editor:
            attacker_username = pi_config["parameters"]["attacker_username"]
            attacker_password = pi_config["parameters"]["attacker_password"]
            reddit_editor.create_user("", "", attacker_username, "", attacker_password)
            issue_url = reddit_editor.create_post_with_title_and_text(
                "allentown",
                "A note from a gentleman thief " + str(task_id),
                pi_config["instantiated_instruction"],
                attacker_username,
            )
            if "user_username" in pi_config["parameters"]:
                user_username = pi_config["parameters"]["user_username"]
                user_password = pi_config["parameters"]["user_password"]
                reddit_editor.login(user_username, user_password)
                if "user_post_title" in pi_config["parameters"]:
                    pi_config["parameters"]["user_post_url"] = (
                        reddit_editor.create_post_with_title_and_text(
                            pi_config["parameters"]["user_post_forum"],
                            pi_config["parameters"]["user_post_title"],
                            pi_config["parameters"]["user_post_text"],
                            user_username,
                        )
                    )
                elif "user_msg_txt" in pi_config["parameters"]:
                    pi_config["parameters"]["user_msg_url"] = reddit_editor.send_message(
                        pi_config["parameters"]["user_msg_recipient"],
                        pi_config["parameters"]["user_msg_txt"],
                    )
                elif "username_to_block" in pi_config["parameters"]:
                    reddit_editor.block_user(user_username, pi_config["parameters"]["username_to_block"])

        user_goal = RedditUserGoals(issue_url)
        return self._create_task_dicts(user_goal, user_goal_idx, task_id, issue_url, pi_config)


@click.command()
@click.option(
    "--config",
    type=str,
    default="configs/experiment_config.raw.json",
    help="Where to find the config for prompt injections",
)
@click.option(
    "--only-environment",
    type=click.Choice(["gitlab", "reddit", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Restrict prompt injection to a single environment (e.g. gitlab only).",
)
@click.option(
    "--gitlab-domain",
    default="none",
    show_default=True,
    help="GitLab domain URL (e.g., http://example.com:port)",
)
@click.option(
    "--reddit-domain",
    default="none",
    show_default=True,
    help="Reddit domain URL (e.g., http://example.com:port)",
)
@click.option(
    "--model",
    type=click.Choice(
        ["gpt-4o", "gpt-4o-mini", "claude-35", "claude-37"], case_sensitive=False
    ),
    default="gpt-4o",
    help="backbone LLM. Available options: gpt-4o, gpt-4o-mini, claude-35, claude-37",
)
@click.option(
    "--system_prompt",
    type=str,
    default="configs/system_prompts/wa_p_som_cot_id_actree_3s.json",
    help="system_prompt for the backbone LLM. Default = VWA's SOM system prompt for GPT scaffolding",
)
@click.option(
    "--user_goal_idx",
    type=int,
    default=0,
    help="which benign user goal to set (default=0)",
)
@click.option(
    "--injection_format",
    type=str,
    default="generic_plain_text",
    help="prompt injection format, currently available: "
    "goal_hijacking_url_injection, goal_hijacking_plain_text, generic_url_injection, generic_plain_text (default)",
)
@click.option(
    "--output-dir",
    type=str,
    default="/tmp",
    help="Folder to store the output configs and commands to run the agent",
)
@click.option(
    "--output-format",
    type=str,
    default="webarena",
    help="Agentic scaffolding to use. Options: webarena (default), gpt_web_tools, claude, pte, beyond_browsing, react_api_web.",
)
@click.option(
    "--skip-environment",
    is_flag=True,
    default=False,
    help="Whether to do a dry run and skip injecting into the environment (for testing purposes). Default is False.",
)
@click.option(
    "--multi-docker",
    is_flag=True,
    default=False,
    help="Generate a parallel run_agent.sh that uses the multi-docker worker pool (pte and beyond_browsing formats only).",
)
def main(
    config,
    only_environment,
    gitlab_domain,
    reddit_domain,
    model,
    system_prompt,
    user_goal_idx,
    injection_format,
    output_dir,
    output_format,
    skip_environment,
    multi_docker,
):
    only_environment = (only_environment or "all").lower()

    if gitlab_domain == "none":  # try to get it from env var
        gitlab_domain = os.environ.get("GITLAB")
    if reddit_domain == "none":
        reddit_domain = os.environ.get("REDDIT")

    experiment_config = load_prompt_injection_config(config)
    prompt_injection_configs = experiment_config["prompt_injections_setup_config"]
    if only_environment != "all":
        prompt_injection_configs = [
            c for c in prompt_injection_configs if c.get("environment") == only_environment
        ]

    system_prompt = os.path.join(os.getcwd(), system_prompt)
    if output_format == "gpt_web_tools":
        print(
            f"No custom system_prompt support for {output_format}, setting it to empty."
        )
        system_prompt = ""
    elif "claude" in model.lower():
        output_format = "claude"
        with open(system_prompt, "r") as claude_agent_config_file:
            claude_agent_configs = json.load(claude_agent_config_file)
            model = claude_agent_configs["model"]
            system_prompt = claude_agent_configs["system_prompt"]

    editor_list = []
    if only_environment in ("all", "gitlab"):
        editor_list.append(GitlabEditor(gitlab_domain))
    if only_environment in ("all", "reddit"):
        editor_list.append(RedditEditor(reddit_domain))

    web_arena_prompt_injector = WebArenaPromptInjector(
        editor_list, prompt_injection_configs
    )

    path_to_agent_script, path_to_instantiated_prompt_injection_config = (
        web_arena_prompt_injector.inject_in_environment(
            output_dir=output_dir,
            output_format=output_format,
            injection_format=injection_format,
            skip_environment=skip_environment,
            system_prompt=system_prompt,
            user_goal_idx=user_goal_idx,
            model=model,
            multi_docker=multi_docker,
        )
    )

    print(f"The agent script was written out to: {path_to_agent_script}")
    print(
        f"The prompt injection config was written out to: {path_to_instantiated_prompt_injection_config}"
    )


if __name__ == "__main__":
    main()
