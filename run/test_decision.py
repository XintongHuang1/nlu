import sys
import os

# Add the project root directory to sys.path to be able to import functionality from core/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.testing import TestCase, Template, DecisionResult
from core.utils import get_model, get_metric
from dataset_assembly import merge_datasets
import pandas as pd
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import datetime
import numpy as np

from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import argparse

############################# 文件格式修改 #############################
# 格式化的开始时间
START_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# References to relevant data directories
# DATASET_FILE_PATH = os.path.join(".", "data", "full_dataset.csv")

DATASET_FILE_PATH = os.path.join(".", "data", "first_1000_rows.csv")
N_data = 1000

# DATASET_FILE_PATH = os.path.join(".", "data", "Anchoring_dataset_first_30_rows.csv")
# N_data = 30


DECISION_RESULTS = os.path.join(".", "data", "decision_results")


def convert_decisions(ids: list[int], decision_results: list[DecisionResult]) -> pd.DataFrame:
    """
    Converts a list of DecisionResult objects into a DataFrame.

    Args:
        ids (list[int]): The ids of the decision results.
        decision_results (list[DecisionResult]): The DecisionResult objects.

    Returns:
        pd.DataFrame: the DataFrame representation of the decision results.
    """

    decision_data = [
        [
            decision_result.MODEL,
            decision_result.TEMPERATURE,
            decision_result.SEED,
            decision_result.TIMESTAMP,
            decision_result.CONTROL_OPTIONS,
            decision_result.CONTROL_OPTION_SHUFFLING,
            decision_result.CONTROL_ANSWER,
            decision_result.CONTROL_EXTRACTION,
            decision_result.CONTROL_DECISION,
            decision_result.TREATMENT_OPTIONS,
            decision_result.TREATMENT_OPTION_SHUFFLING,
            decision_result.TREATMENT_ANSWER,
            decision_result.TREATMENT_EXTRACTION,
            decision_result.TREATMENT_DECISION,
            decision_result.STATUS,
            decision_result.ERROR_MESSAGE
        ]
        for decision_result in decision_results
    ]
    
    # Wrap the results in a new DataFrame
    decision_df = pd.DataFrame({
        "id": ids,
        "model": list(zip(*decision_data))[0],
        "temperature": list(zip(*decision_data))[1],
        "seed": list(zip(*decision_data))[2],
        "timestamp": list(zip(*decision_data))[3],
        "control_options": list(zip(*decision_data))[4],
        "control_option_order": list(zip(*decision_data))[5],
        "control_answer": list(zip(*decision_data))[6],
        "control_extraction": list(zip(*decision_data))[7],
        "control_decision": list(zip(*decision_data))[8],
        "treatment_options": list(zip(*decision_data))[9],
        "treatment_option_order": list(zip(*decision_data))[10],
        "treatment_answer": list(zip(*decision_data))[11],
        "treatment_extraction": list(zip(*decision_data))[12],
        "treatment_decision": list(zip(*decision_data))[13],
        "status": list(zip(*decision_data))[14],
        "error_message": list(zip(*decision_data))[15]
    })

    return decision_df

################ PRPOMPT STRATEGY ####################### 加入 prompt_strategy变量
def decide_batch(batch: pd.DataFrame, model_name: str, randomly_flip_options: bool, shuffle_answer_options: bool, temperature: float, seed: int, 
                prompt_strategy, progress_bar=None):
    """
    Decides the dataset batch using the specified model.

    Args:
        batch (pd.DataFrame): The batch of the dataset with generated test case instances to decide.
        model_name (str): The name of the model to use for obtaining decisions.
        randomly_flip_options (bool): Whether to reverse the answer options in 50% of test cases.
        shuffle_answer_options (bool): Whether to shuffle the answer options randomly for all test cases.
        temperature (float): The temperature to use for the decision model.
        seed (int): The seed to use for reproducibility.
    """

    # Get an instance of the model
    model = get_model(model_name, randomly_flip_options=randomly_flip_options, shuffle_answer_options=shuffle_answer_options)

    # Initialize a decision batch
    decision_batch = None

    # Identify the biases in the batch
    biases = [''.join(' ' + char if char.isupper() else char for char in bias).strip().title().replace(' ', '') for bias in batch["bias"].unique()]

    # Iterate over all biases in the batch
    for bias in biases:
        test_cases, ids = [], []

        # Construct test cases for all relevant rows in the batch
        for _, row in batch[batch['bias'].str.strip().str.title().str.replace(' ', '') == bias].iterrows():
            ids.append(row['id'])
            
################ PRPOMPT STRATEGY ####################### 0.原始版本，保存以便修改
#             test_cases.append(
#                 TestCase(
#                     bias=row["bias"],
#                     control=Template(row["raw_control"]),
#                     treatment=Template(row["raw_treatment"]),
#                     generator=row["generator"],
#                     temperature=row["temperature"],
#                     seed=row["seed"],
#                     scenario=row["scenario"],
#                     variant=row["variant"],
#                     remarks=row["remarks"],
#                 )
#             )
################ PRPOMPT STRATEGY ####################### 1.构造 Template 对象（基于 XML 字符串）
            control_template = Template(row["raw_control"])
            treatment_template = Template(row["raw_treatment"])
            
################ PRPOMPT STRATEGY ####################### 2.提取原始 <prompt> 内容
            control_prompt = control_template._data.find("prompt").text
            treatment_prompt = treatment_template._data.find("prompt").text
            
################ PRPOMPT STRATEGY ####################### 3.使用 apply_prompt_strategy 拼接策略提示
            updated_control_prompt = apply_prompt_strategy(control_prompt, prompt_strategy)
            updated_treatment_prompt = apply_prompt_strategy(treatment_prompt, prompt_strategy)
            
################ PRPOMPT STRATEGY ####################### 更新模板中的 <prompt> 内容
            control_template._data.find("prompt").text = updated_control_prompt
            treatment_template._data.find("prompt").text = updated_treatment_prompt
################ PRPOMPT STRATEGY ####################### 添加到测试集
            test_cases.append(
                TestCase(
                    bias=row["bias"],
                    control=control_template,
                    treatment=treatment_template,
                    generator=row["generator"],
                    temperature=row["temperature"],
                    seed=row["seed"],
                    scenario=row["scenario"],
                    variant=row["variant"],
                    remarks=row["remarks"],
                )
            )

        # Decide the test cases and obtain the DecisionResult objects
        decision_results = model.decide_all(test_cases, temperature, seed, max_retries=1, progress_bar=progress_bar)

        # Store all the results (both failed and completed) in a new DataFrame
        decision_df = convert_decisions(ids, decision_results)

################ 3.PRPOMPT STRATEGY ####################### 添加 prompt_strategy 信息
        # 添加 prompt_strategy 信息
        decision_df["prompt_strategy"] = prompt_strategy

        # Get indices of the decisions that failed: they have status "ERROR"
        failed_idx = [i for i, decision_result in enumerate(decision_results) if decision_result.STATUS == "ERROR"]

        # Remove failed decisions from the decision results to calculate the metric
        decision_results = [decision_result for i, decision_result in enumerate(decision_results) if i not in failed_idx]

        # Remove corresponding test cases to calculate the metric
        test_cases = [test_case for i, test_case in enumerate(test_cases) if i not in failed_idx]

        # Calculate the metrics if we have any correct decisions
        if len(test_cases) > 0 and len(decision_results) > 0:
            metric = get_metric(bias)(test_results=list(zip(test_cases, decision_results)))
            individual_scores = metric.compute()

            # Store the results and weights in the rows of the "OK" decisions
            decision_df.loc[decision_df['status'] == "OK", "individual_score"] = individual_scores
            decision_df.loc[decision_df['status'] == "OK", "weight"] = metric.test_weights
        decision_df.loc[:, "bias"] = bias

        # Append this bias's decisions to the overall decisions for the batch
        decision_batch = (
            decision_df
            if decision_batch is None
            else pd.concat([decision_batch, decision_df], ignore_index=True)
        )
    
    # Save the decisions for the batch to a CSV file with a unique name based on the process ID and timestamp
############################# 文件格式修改 #############################
    model_time_file = f"{safe_filename(model_name)}_{START_TIME}"
    file_name = os.path.join(DECISION_RESULTS, model_time_file, f"batch_{os.getpid()}_decided_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    decision_batch.to_csv(file_name, index=False)
    # file_name = f"batch_{os.getpid()}_decided_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    # file_path = os.path.join(DECISION_RESULTS, safe_filename(model_name), safe_filename(file_name))
    # decision_batch.to_csv(file_path, index=False)

import re

def safe_filename(name):
    """将路径中的非法字符替换为下划线"""
    return re.sub(r'[<>:"|?*]', '_', name)

################ PRPOMPT STRATEGY ####################### 加入 prompt_strategy变量
def decide_dataset(dataset: pd.DataFrame, model_name: str, n_batches: int, n_workers: int, randomly_flip_options: bool, shuffle_answer_options: bool, temperature: float, seed: int
                   ,prompt_strategy: str):
    """
    Function that encapsulates the parallel decision-making process for a dataset.

    Args:
        dataset (pd.DataFrame): The dataset with generated test case instances to decide.
        model_name (str): The name of the model to use for obtaining decisions.
        n_batches (int): The number of equally-sized batches to split the dataset into for distribution across the parallel workers.
        n_workers (int): The maximum number of parallel workers used.
        randomly_flip_options (bool): Whether to reverse the answer options in 50% of test cases.
        shuffle_answer_options (bool): Whether to shuffle the answer options randomly for all test cases.
        temperature (float): The temperature to use for the decision model.
        seed (int): The seed to use for reproducibility.
    """

    # Prepare the directory to store the decision results of the model
############################# 文件格式修改 #############################
    model_time_file = f"{safe_filename(model_name)}_{START_TIME}"
    results_directory = os.path.join(DECISION_RESULTS, model_time_file)
    os.makedirs(results_directory, exist_ok=True)

    # Split the dataset into equally-sized batches for distribution across the parallel workers
    batches = np.array_split(dataset, n_batches)
    
    # Allocate the batches to the workers to obtain decisions.
    # with tqdm(total=len(batches)) as progress_bar:
    # with tqdm(total=len(batches), desc="🚀 Processing batches") as progress_bar:
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executer:
    #         for _ in executer.map(
    #             partial(
    #                 decide_batch,
    #                 model_name=model_name,
    #                 randomly_flip_options=randomly_flip_options,
    #                 shuffle_answer_options=shuffle_answer_options,
    #                 temperature=temperature,
    #                 seed=seed
    #             ),
    #             batches
    #         ):
    #             progress_bar.update()


    # decide_batch_partial = partial(
    #     decide_batch,
    #     model_name=model_name,
    #     randomly_flip_options=randomly_flip_options,
    #     shuffle_answer_options=shuffle_answer_options,
    #     temperature=temperature,
    #     seed=seed
    # )

    # process_map(
    #     decide_batch_partial,
    #     batches,
    #     max_workers=n_workers,
    #     desc="🚀 Processing batches"
    # )
    
    total_test_cases = len(dataset)

    if n_workers == 1:
        with tqdm(total=total_test_cases, desc="🧠 Deciding test cases") as progress_bar:
            for batch in batches:
                decide_batch(
                    batch=batch,
                    model_name=model_name,
                    randomly_flip_options=randomly_flip_options,
                    shuffle_answer_options=shuffle_answer_options,
                    temperature=temperature,
                    seed=seed,
################ PRPOMPT STRATEGY ####################### 输入 prompt_strategy变量
                    prompt_strategy=prompt_strategy,
                    progress_bar=progress_bar
                )
    else:
        from functools import partial
        from tqdm.contrib.concurrent import process_map

        decide_batch_partial = partial(
            decide_batch,
            model_name=model_name,
            randomly_flip_options=randomly_flip_options,
            shuffle_answer_options=shuffle_answer_options,
            temperature=temperature,
################ PRPOMPT STRATEGY ####################### 输入 prompt_strategy变量
            prompt_strategy=prompt_strategy,
            seed=seed
        )

        process_map(
            decide_batch_partial,
            batches,
            max_workers=n_workers,
            desc="🚀 Processing batches"
        )



    # Merge all batch results into a single CSV containing all decision results of the model
    merge_datasets(results_directory, DECISION_RESULTS, f"{safe_filename(model_name)}_{prompt_strategy}_{N_data}_seed{seed}_{START_TIME}.csv", add_id=False)

################ PRPOMPT STRATEGY ####################### apply_prompt_strategy函数
# def apply_prompt_strategy(base_prompt: str, strategy: str) -> str:
#     # 格式统一提示（无解释性干预）
#     format_ending = "\nPlease ensure your final response ends with the format: Option X"

#     if strategy == "zero-shot":
#         return base_prompt + format_ending

# ################ PRPOMPT STRATEGY ####################### 这些例子来自于ID: 433, 912, 845
#     elif strategy == "few-shot":
#         few_shot_examples = (
#             "Example 1:\n"
#             "Situation 1: You are a quality control manager. The prompt asks: 'Do you intend to allocate more than 49% for quality approval?'\n"
#             "Situation 2: You are asked to choose a percentage of products that must meet quality standards to approve a batch (no number suggested).\n"
#             "Note: The number in Situation 1 may act as an anchor.\n"
#             "Instruction: These two situations describe the same decision. Your answer should be the same.\n\n"

#             "Example 2:\n"
#             "Situation 1: You are deciding how much support time to allocate to a new ticketing system (no hints).\n"
#             "Situation 2: The prompt includes: 'Do you intend to allocate more than 58%?'\n"
#             "Note: Situation 2 introduces a numerical anchor.\n"
#             "Instruction: Your decision should be based on operational needs, not the number. Answer consistently.\n\n"

#             "Example 3:\n"
#             "Situation 1: You are a claims manager allocating reserve funds for a disputed insurance case.\n"
#             "Situation 2: The prompt adds: 'Do you intend to allocate more than 41%?'\n"
#             "Note: The phrasing in Situation 2 may cause anchoring bias.\n"
#             "Instruction: Both situations require the same judgment. Provide the same answer.\n\n"

#             "Now consider the following situation:\n"
#         )
#         return few_shot_examples + base_prompt + format_ending

#     elif strategy == "cot":
#         return (
#             base_prompt
#             + "\nPlease think step by step before answering."
#             + format_ending
#         )

#     elif strategy == "pot":
#         return (
#             base_prompt
#             + "\nDescribe your decision-making process before choosing the answer."
#             + format_ending
#         )

#     elif strategy == "reflection":
#         return (
#             base_prompt
#             + "\nBefore finalizing your answer, reflect on whether any Anchoring Biases might have influenced your choice."
#             + format_ending
#         )
#     elif strategy == "debias-rewrite":
#         return (
#             base_prompt
#             + "\n\nFirst, identify whether the following question contains any anchoring cues (such as specific numbers or suggestive phrasing)."
#             + "\nThen, rewrite the question in a neutral way that removes any anchoring influence."
#             + "\nFinally, answer the rewritten question with your chosen option."
#             + "\nPlease ensure your final response ends with the format: Option X"
#         )
#     else:
#         return base_prompt + format_ending  # fallback

################ PRPOMPT STRATEGY ####################### apply_prompt_strategy函数
def apply_prompt_strategy(base_prompt: str, strategy: str) -> str:
    # 格式统一提示（无解释性干预）
    format_ending = "\nPlease ensure your final response ends with the format: Option X"

    # ✅ 支持组合策略，先拆解
    allowed_components = {"zero-shot", "few-shot", "cot", "pot", "reflection", "debias-rewrite"}
    components = strategy.split("+")
    for comp in components:
        if comp not in allowed_components:
            raise ValueError(f"[Prompt Strategy Error] Unsupported prompt strategy component: '{comp}'. "
                             f"Allowed: {', '.join(sorted(allowed_components))}")

    # ✅ 初始化 prompt 内容
    prompt = base_prompt

    # ✅ few-shot 例子模板（只在需要时注入）
    few_shot_examples = (
        "Example 1:\n"
        "Situation 1: You are a quality control manager. The prompt asks: 'Do you intend to allocate more than 49% for quality approval?'\n"
        "Situation 2: You are asked to choose a percentage of products that must meet quality standards to approve a batch (no number suggested).\n"
        "Note: The number in Situation 1 may act as an anchor.\n"
        "Instruction: These two situations describe the same decision. Your answer should be the same.\n\n"

        "Example 2:\n"
        "Situation 1: You are deciding how much support time to allocate to a new ticketing system (no hints).\n"
        "Situation 2: The prompt includes: 'Do you intend to allocate more than 58%?'\n"
        "Note: Situation 2 introduces a numerical anchor.\n"
        "Instruction: Your decision should be based on operational needs, not the number. Answer consistently.\n\n"

        "Example 3:\n"
        "Situation 1: You are a claims manager allocating reserve funds for a disputed insurance case.\n"
        "Situation 2: The prompt adds: 'Do you intend to allocate more than 41%?'\n"
        "Note: The phrasing in Situation 2 may cause anchoring bias.\n"
        "Instruction: Both situations require the same judgment. Provide the same answer.\n\n"

        "Now consider the following situation:\n"
    )

    # ✅ 应用每个子策略逻辑
    for comp in components:
        if comp == "zero-shot":
            continue  # zero-shot 不加额外内容
        elif comp == "few-shot":
            prompt = few_shot_examples + prompt
        elif comp == "cot":
            prompt += "\nPlease think step by step before answering."
        elif comp == "pot":
            prompt += "\nDescribe your decision-making process before choosing the answer."
        elif comp == "reflection":
            prompt += "\nBefore finalizing your answer, reflect on whether any Anchoring Biases might have influenced your choice."
        elif comp == "debias-rewrite":
            prompt += (
                "\n\nFirst, identify whether the following question contains any anchoring cues (such as specific numbers or suggestive phrasing)."
                + "\nThen, rewrite the question in a neutral way that removes any anchoring influence."
                + "\nFinally, answer the rewritten question with your chosen option."
            )

    return prompt + format_ending


def main():
    """
    The main function of this script that obtains decisions from models for generated test case instances.
    """

    # Define a command line argument parser
    parser = argparse.ArgumentParser(description="This script obtains decisions from models for generated test case instances.")
################ PRPOMPT STRATEGY ####################### 命令行输入prompt_strategy变量
    # parser.add_argument(
    # "--prompt_strategy",
    # type=str,
    # choices=["zero-shot", "few-shot", "cot", "pot", "reflection","debias-rewrite"],
    # default="zero-shot",
    # help="The prompt engineering strategy to apply")
    parser.add_argument(
    "--prompt_strategy",
    type=str,
    default="zero-shot",
    help="The prompt engineering strategy to apply. You can use '+' to combine strategies (e.g., 'few-shot+reflection').")
    parser.add_argument("--dataset", type=str, help="The path to the dataset file with the test case instances.", default=DATASET_FILE_PATH)
    parser.add_argument("--model", type=str, help="The name of the model to obtain decisions from.", default="GPT-4o-Mini")
    parser.add_argument("--n_workers", type=int, help="The maximum number of parallel workers obtaining decisions from the model.", default=100)
    parser.add_argument("--n_batches", type=int, help="The number of equally-sized batches to split the dataset into to distribute them to the workers.", default=3000)
    parser.add_argument("--temperature", type=float, help="Temperature value of the decision model", default=0.0)
    parser.add_argument("--seed", type=int, help="The seed to use for reproducibility.", default=42)
    args = parser.parse_args()

    # Load the dataset with generated test case instances
    dataset = pd.read_csv(args.dataset)

    # Obtain decisions from the model for all test case instances in the dataset
    print(f"Starting the decision-making process with {args.n_workers} parallel workers ...")
    start_time = datetime.datetime.now()
    decide_dataset(
        dataset=dataset,
        model_name=args.model,
        n_batches=args.n_batches,
        n_workers=args.n_workers,
        randomly_flip_options=True,
        shuffle_answer_options=False,
        temperature=0.0,
        seed=42,
################ PRPOMPT STRATEGY ####################### 输入prompt_strategy变量
        prompt_strategy=args.prompt_strategy
    )
    print(f"All decisions obtained from model '{args.model}' in {datetime.datetime.now() - start_time} seconds.")


if __name__ == "__main__":
    main()
