#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Prerequisite 1: python virtual environment and built Docker container (TODO: describe how to do these in README)
# Prerequisite 2: set the required authentication environment variables (AZURE_API_ENDPOINT and AZURE_API_KEY for GPT/WebArena and AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN for Claude)

set -e

# Load environment variables from repo-root .env if present
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENV_FILE="$SCRIPT_DIR/../../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

export OUTPUT_DIR=${1:-/tmp/computer-use-agent-logs/}
export MODEL=${2:-gpt-4o}
export SYSTEM_PROMPT=${3:-configs/system_prompts/wa_p_som_cot_id_actree_3s.json}
export CONFIG_PATH=${4:-configs/experiment_config.raw.json}
export USER_GOAL_IDX=${5:-0}
export INJECTION_FORMAT=${6:-goal_hijacking_plain_text}
export OUTPUT_FORMAT=${7:-webarena}
export MULTI_DOCKER_FLAG=${8:-}

if [[ "${OUTPUT_DIR}" != */ ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/"
fi

if [ -d "$OUTPUT_DIR" ]; then
  echo "Deleting existing OUTPUT_DIR=${OUTPUT_DIR}"
  rm -rf "$OUTPUT_DIR"
fi

echo "Creating the new OUTPUT_DIR=${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

# Normalize to an absolute path (with trailing slash) so later steps that `cd` into
# sibling repos (e.g. visualwebarena for the final-step evaluator) still resolve the
# per-run output directory correctly.
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")/"
echo "Resolved OUTPUT_DIR=${OUTPUT_DIR}"

# Note: evaluators now write step-by-step ASR and attacker/user utility JSON directly
# into OUTPUT_DIR (self-contained per run); nothing lands in /tmp, so there is no
# cross-run /tmp state left to clean up here.

echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "GITLAB_DOMAIN: $GITLAB"
echo "REDDIT_DOMAIN: $REDDIT"
echo "Model: $MODEL"
echo "SYSTEM_PROMPT: $SYSTEM_PROMPT"
echo "USER_GOAL_IDX: $USER_GOAL_IDX"
echo "INJECTION_FORMAT: $INJECTION_FORMAT"
echo "OUTPUT_FORMAT: $OUTPUT_FORMAT"
echo "MULTI_DOCKER_FLAG: $MULTI_DOCKER_FLAG"

##### STEP 1: Inject prompts and create tasks in web environment ######
echo "SCRIPT_DIR: $SCRIPT_DIR"
cd $SCRIPT_DIR/..
cp $CONFIG_PATH "${OUTPUT_DIR}experiment_config.json"
echo "step 1 | preparing prompt injections and tasks..."
source venv/bin/activate
python prompt_injector.py --config $CONFIG_PATH \
                          --gitlab-domain $GITLAB \
                          --reddit-domain $REDDIT \
                          --model $MODEL \
                          --system_prompt $SYSTEM_PROMPT \
                          --output-dir $OUTPUT_DIR \
                          --user_goal_idx $USER_GOAL_IDX \
                          --injection_format $INJECTION_FORMAT \
                          --output-format $OUTPUT_FORMAT \
                          $MULTI_DOCKER_FLAG
deactivate
# bash step1_setup_prompt_injections.sh $OUTPUT_DIR $MODEL $SYSTEM_PROMPT $CONFIG_PATH $USER_GOAL_IDX $INJECTION_FORMAT $OUTPUT_FORMAT
##### -----------


##### STEP 2: Run agents on tasks ######
echo "SCRIPT_DIR: $SCRIPT_DIR"
AGENT_RUN_SCRIPT="${OUTPUT_DIR}run_agent.sh"
echo "step 2 | Executing agent script at $AGENT_RUN_SCRIPT"
chmod -R 777 $OUTPUT_DIR
if [ "$OUTPUT_FORMAT" = "pte" ] || [ "$OUTPUT_FORMAT" = "beyond_browsing" ] || [ "$OUTPUT_FORMAT" = "react_api_web" ] || [ "$OUTPUT_FORMAT" = "react_web" ] || [ "$OUTPUT_FORMAT" = "claude_code" ] || [ "$OUTPUT_FORMAT" = "antigravity" ] || [ "$OUTPUT_FORMAT" = "ucm_agent" ] || [ "$OUTPUT_FORMAT" = "codex" ]; then
    bash "$AGENT_RUN_SCRIPT"
else
    cd $SCRIPT_DIR/../../visualwebarena/
    source venv/bin/activate
    bash "$AGENT_RUN_SCRIPT"
    # bash step2_run_agent.sh $OUTPUT_DIR
    deactivate
fi
##### -----------


##### STEP 3: Run evaluations ######
echo "SCRIPT_DIR: $SCRIPT_DIR"
cd $SCRIPT_DIR/..
LOG_DIR="${OUTPUT_DIR}agent_logs/"
TASK_DIR="${OUTPUT_DIR}webarena_tasks/"
ATTACKER_TASK_DIR="${OUTPUT_DIR}webarena_tasks_attacker/"
echo "step 3 | OUTPUT_DIR: $OUTPUT_DIR"
echo "step 3 | OUTPUT_FORMAT: $OUTPUT_FORMAT"

# Map pte/beyond_browsing formats to gpt_web_tools for evaluators (same JSONL format)
EVAL_FORMAT=$OUTPUT_FORMAT
if [ "$OUTPUT_FORMAT" = "pte" ] || [ "$OUTPUT_FORMAT" = "beyond_browsing" ] || [ "$OUTPUT_FORMAT" = "react_api_web" ] || [ "$OUTPUT_FORMAT" = "react_web" ] || [ "$OUTPUT_FORMAT" = "claude_code" ] || [ "$OUTPUT_FORMAT" = "antigravity" ] || [ "$OUTPUT_FORMAT" = "ucm_agent" ] || [ "$OUTPUT_FORMAT" = "codex" ]; then
    EVAL_FORMAT="gpt_web_tools"
fi

# first report ASR (also persists per-task verdicts into the sub-run dir)
source venv/bin/activate
python evaluator_step_by_step.py --log-folder $LOG_DIR --task-folder $TASK_DIR --attacker-task-folder $ATTACKER_TASK_DIR --format $EVAL_FORMAT --output-dir "$OUTPUT_DIR"
echo "Done evaluating attack goals step-by-step, above score is ASR!"
deactivate

# next report user task success rate
cd ../visualwebarena/
source venv/bin/activate
bash prepare.sh
# evaluate attacker task performance (i.e. if final goal of the attacker was achieved).
# Persist into the sub-run dir (not /tmp, which gets clobbered across grid cells) with a
# per-task breakdown, so attacker/user utility is retraceable per template.
python evaluator_final_step.py --log-folder $LOG_DIR --task-folder $ATTACKER_TASK_DIR --format $EVAL_FORMAT --output-file "${OUTPUT_DIR}attacker_utility.json"
echo "Done evaluating attacker goals, above score is ASR!"
# evaluate user task performance
python evaluator_final_step.py --log-folder $LOG_DIR --task-folder $TASK_DIR --output-file "${OUTPUT_DIR}user_utility.json"
echo "Done evaluating user objectives!"
deactivate

##### STEP 3.5: Consolidate per-template results ######
echo "step 3.5 | Consolidating per-template results for $OUTPUT_DIR"
cd $SCRIPT_DIR/..
source venv/bin/activate
# Non-fatal: a consolidation hiccup must not skip Step 4 (environment cleanup). run.py
# also runs collect_results itself, so the table is still produced when driven from there.
python collect_results.py --run-dir "$OUTPUT_DIR" || echo "step 3.5 | WARNING: collect_results failed (continuing)"
deactivate
##### -----------
# bash step3_run_evaluators.sh $OUTPUT_DIR $OUTPUT_FORMAT
##### -----------


##### STEP 4: Cleanup environment ######
echo "SCRIPT_DIR: $SCRIPT_DIR"
cd $SCRIPT_DIR/..
PROMPT_INJECTION_CONFIG="${OUTPUT_DIR}instantiated_prompt_injections_config.json"
echo "step 4 | OUTPUT_DIR: $OUTPUT_DIR"
echo "step 4 | PROMPT_INJECTION_CONFIG: $PROMPT_INJECTION_CONFIG"
source venv/bin/activate
python environment_cleanup.py --prompt-injection-config-path "$PROMPT_INJECTION_CONFIG" --gitlab-domain $GITLAB --reddit-domain $REDDIT $MULTI_DOCKER_FLAG --worker-ids-dir "${OUTPUT_DIR}worker_ids/"
deactivate
# bash step4_cleanup.sh $OUTPUT_DIR
##### -----------