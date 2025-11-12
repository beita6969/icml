#!/usr/bin/env python3
"""
Convert AFlow dataset format to ROLL format with messages field.
"""
import json
import sys
from pathlib import Path


def convert_math_to_roll(aflow_data):
    """Convert MATH dataset from AFlow format to ROLL format."""
    return {
        "id": str(hash(aflow_data["problem"])),  # Generate unique ID
        "domain": "math",
        "tag": "math",
        "difficulty": 0.5,  # Default difficulty
        "prompt": aflow_data["problem"],
        "messages": json.dumps([
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\\\boxed{}."},
            {"role": "user", "content": aflow_data["problem"]}
        ]),
        "ground_truth": aflow_data["solution"],
        # Standardized optional fields
        "level": aflow_data.get("level", "Unknown"),
        "type": aflow_data.get("type", "Math"),
        "cot": "",
        "entry_point": "",
        "test": "",
        "test_list": "",
        "context": "",
    }


def convert_gsm8k_to_roll(aflow_data):
    """Convert GSM8K dataset from AFlow format to ROLL format."""
    return {
        "id": aflow_data.get("id", str(hash(aflow_data["question"]))),
        "domain": "gsm8k",
        "tag": "gsm8k",
        "difficulty": 0.5,
        "prompt": aflow_data["question"],
        "messages": json.dumps([
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\\\boxed{}."},
            {"role": "user", "content": aflow_data["question"]}
        ]),
        "ground_truth": aflow_data["answer"],
        # Standardized optional fields
        "level": "",
        "type": "",
        "cot": aflow_data.get("cot", ""),
        "entry_point": "",
        "test": "",
        "test_list": "",
        "context": "",
    }


def convert_humaneval_to_roll(aflow_data):
    """Convert HumanEval dataset from AFlow format to ROLL format."""
    problem_text = aflow_data.get("prompt", "") or aflow_data.get("question", "")
    return {
        "id": aflow_data.get("task_id", aflow_data.get("id", str(hash(problem_text)))),
        "domain": "code_human",
        "tag": "code_human",
        "difficulty": 0.5,
        "prompt": problem_text,
        "messages": json.dumps([
            {"role": "system", "content": "Please provide a complete Python function implementation."},
            {"role": "user", "content": problem_text}
        ]),
        "ground_truth": aflow_data.get("canonical_solution", ""),
        # Standardized optional fields
        "level": "",
        "type": "",
        "cot": "",
        "entry_point": aflow_data.get("entry_point", ""),
        "test": aflow_data.get("test", ""),
        "test_list": "",
        "context": "",
    }


def convert_mbpp_to_roll(aflow_data):
    """Convert MBPP dataset from AFlow format to ROLL format."""
    problem_text = aflow_data.get("text", aflow_data.get("question", ""))
    return {
        "id": str(aflow_data.get("task_id", aflow_data.get("id", hash(problem_text)))),
        "domain": "code_mbpp",
        "tag": "code_mbpp",
        "difficulty": 0.5,
        "prompt": problem_text,
        "messages": json.dumps([
            {"role": "system", "content": "Please provide a complete Python function implementation."},
            {"role": "user", "content": problem_text}
        ]),
        "ground_truth": aflow_data.get("code", ""),
        # Standardized optional fields
        "level": "",
        "type": "",
        "cot": "",
        "entry_point": "",
        "test": "",
        "test_list": json.dumps(aflow_data.get("test_list", [])),
        "context": "",
    }


def convert_hotpotqa_to_roll(aflow_data):
    """Convert HotpotQA dataset from AFlow format to ROLL format."""
    question = aflow_data.get("question", "")
    return {
        "id": aflow_data.get("id", str(hash(question))),
        "domain": "qa_hotpot",
        "tag": "qa_hotpot",
        "difficulty": 0.5,
        "prompt": question,
        "messages": json.dumps([
            {"role": "system", "content": "Please answer the question based on the given context."},
            {"role": "user", "content": question}
        ]),
        "ground_truth": aflow_data.get("answer", ""),
        # Standardized optional fields
        "level": "",
        "type": "",
        "cot": "",
        "entry_point": "",
        "test": "",
        "test_list": "",
        "context": json.dumps(aflow_data.get("context", "")),
    }


def convert_drop_to_roll(aflow_data):
    """Convert DROP dataset from AFlow format to ROLL format."""
    passage = aflow_data.get("passage", "")
    question = aflow_data.get("question", "")
    full_text = f"Passage: {passage}\n\nQuestion: {question}"

    return {
        "id": aflow_data.get("query_id", str(hash(question))),
        "domain": "qa_drop",
        "tag": "qa_drop",
        "difficulty": 0.5,
        "prompt": full_text,
        "messages": json.dumps([
            {"role": "system", "content": "Please answer the question based on the given passage."},
            {"role": "user", "content": full_text}
        ]),
        "ground_truth": aflow_data.get("answer", ""),
        # Standardized optional fields
        "level": "",
        "type": "",
        "cot": "",
        "entry_point": "",
        "test": "",
        "test_list": "",
        "context": "",
    }


CONVERTERS = {
    "math": convert_math_to_roll,
    "gsm8k": convert_gsm8k_to_roll,
    "humaneval": convert_humaneval_to_roll,
    "mbpp": convert_mbpp_to_roll,
    "hotpotqa": convert_hotpotqa_to_roll,
    "drop": convert_drop_to_roll,
}


def convert_dataset(input_file: str, output_file: str, dataset_type: str):
    """Convert a dataset from AFlow format to ROLL format."""
    converter = CONVERTERS.get(dataset_type)
    if not converter:
        print(f"Error: Unknown dataset type '{dataset_type}'")
        print(f"Available types: {', '.join(CONVERTERS.keys())}")
        sys.exit(1)

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Converting {dataset_type} dataset: {input_file} -> {output_file}")

    converted_count = 0
    error_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                aflow_data = json.loads(line)
                roll_data = converter(aflow_data)
                outfile.write(json.dumps(roll_data, ensure_ascii=False) + '\n')
                converted_count += 1
            except Exception as e:
                error_count += 1
                print(f"Warning: Error on line {line_num}: {e}")
                if error_count > 10:
                    print("Too many errors, stopping...")
                    sys.exit(1)

    print(f"✅ Converted {converted_count} examples")
    if error_count > 0:
        print(f"⚠️  {error_count} errors encountered")

    return converted_count


def main():
    """Convert all AFlow datasets to ROLL format."""
    aflow_data_dir = Path("/home/username/AFlow/data/datasets")
    roll_data_dir = Path("/home/username/ROLL/examples/aflow_integration/data")

    # Create output directory
    roll_data_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("math", "math_validate.jsonl"),
        ("gsm8k", "gsm8k_validate.jsonl"),
        ("humaneval", "humaneval_validate.jsonl"),
        ("mbpp", "mbpp_validate.jsonl"),
        ("hotpotqa", "hotpotqa_validate.jsonl"),
        ("drop", "drop_validate.jsonl"),
    ]

    print("=" * 80)
    print("AFlow to ROLL Dataset Conversion")
    print("=" * 80)

    total_converted = 0
    for dataset_type, filename in datasets:
        input_file = aflow_data_dir / filename
        output_file = roll_data_dir / filename

        if input_file.exists():
            count = convert_dataset(str(input_file), str(output_file), dataset_type)
            total_converted += count
            print()
        else:
            print(f"⚠️  Skipping {filename} (not found)")
            print()

    print("=" * 80)
    print(f"✅ Total converted: {total_converted} examples")
    print(f"📁 Output directory: {roll_data_dir}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single file conversion mode
        if len(sys.argv) != 4:
            print("Usage: python convert_aflow_to_roll.py <input_file> <output_file> <dataset_type>")
            print(f"Dataset types: {', '.join(CONVERTERS.keys())}")
            sys.exit(1)

        convert_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        # Batch conversion mode
        main()
