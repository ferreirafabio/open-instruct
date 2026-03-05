#!/usr/bin/env python3
"""
Download and preprocess SFT datasets for OLMo-core training.

This script:
1. Downloads datasets from HuggingFace Hub
2. Applies transform functions to convert to standard `messages` format
3. Saves each dataset as a separate parquet file

Usage:
    # Download and preprocess all datasets
    python preprocess_datasets.py

    # Preprocess specific mixtures
    python preprocess_datasets.py --mixtures nvidia-Nemotron mlabonne-open-perfectblend

    # Dry run (show what would be processed)
    python preprocess_datasets.py --dry-run

    # List available mixtures
    python preprocess_datasets.py --list-mixtures

Output:
    Each dataset is saved as: data/datasets_mixture_sft_preprocessed/<mixture-name>.parquet
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192 * 1024), b""):  # 8MB chunks
            h.update(chunk)
    return h.hexdigest()


def verify_file_hash(file_path: Path, expected_hash: str, algorithm: str = "sha256") -> bool:
    """Verify file hash matches expected value."""
    if not file_path.exists():
        return False
    actual_hash = compute_file_hash(file_path, algorithm)
    return actual_hash == expected_hash


################################################################################
# Transform functions (from generate.py, made executable)
################################################################################


def transform_smoltalk2(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms examples from the HuggingFaceTB/smoltalk2 dataset for SFT.
    - Merges consecutive turns from the same role.
    - Excludes examples that contain any tools.
    """
    chat_template_kwargs = example.get("chat_template_kwargs", {})
    if (
        chat_template_kwargs.get("python_tools")
        and len(chat_template_kwargs["python_tools"]) > 0
    ) or (
        chat_template_kwargs.get("xml_tools")
        and len(chat_template_kwargs["xml_tools"]) > 0
    ):
        return {"messages": []}

    messages = [example["messages"][0]]
    for message in example["messages"][1:]:
        if message["role"] == messages[-1]["role"]:
            messages[-1]["content"] += "\n" + message["content"]
        else:
            messages.append(message)

    return {"messages": messages}


def transform_perfectblend(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms examples from the mlabonne/open-perfectblend dataset for SFT.
    """
    conversations = example["conversations"]
    if not conversations or not isinstance(conversations, list):
        return {"messages": []}

    role_map = {"human": "user", "gpt": "assistant"}
    messages = []
    for turn in conversations:
        role = turn["from"]
        content = turn["value"]
        current_role = role_map.get(role, role)
        current_content = content.strip()
        if messages and messages[-1]["role"] == current_role:
            messages[-1]["content"] += "\n" + current_content
        else:
            messages.append({"role": current_role, "content": current_content})

    return {"messages": messages}


def transform_orca_agentinstruct_1m(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms examples from the microsoft/orca-agentinstruct-1m-v1 dataset.
    """
    return {"messages": json.loads(example["messages"])}


def transform_anthropic_hh_rlhf(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms Anthropic/hh-rlhf text blobs into messages.
    """
    text = example["chosen"]
    pattern = r"(Human|Assistant):([\s\S]+?)(?=(?:Human|Assistant):|$)"
    matches = re.findall(pattern, text)

    messages = []
    role_map = {"Human": "user", "Assistant": "assistant"}

    for role, content in matches:
        clean_role = role_map.get(role, role)
        clean_content = content.strip()
        if messages and messages[-1]["role"] == clean_role:
            messages[-1]["content"] += "\n" + clean_content
        else:
            messages.append({"role": clean_role, "content": clean_content})

    return {"messages": messages}


def transform_stanfordnlp_shp(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms stanfordnlp/SHP examples for SFT.
    """
    if example["labels"] == 1:
        chosen_response = example["human_ref_A"]
    else:
        chosen_response = example["human_ref_B"]

    return {
        "messages": [
            {"role": "user", "content": example["history"]},
            {"role": "assistant", "content": chosen_response},
        ]
    }


def transform_berkeley_nest_nectar(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms berkeley-nest/Nectar examples for SFT.
    """
    raw_prompt = example["prompt"]
    clean_prompt = re.sub(r"^\s*Human:\s*", "", raw_prompt, flags=re.IGNORECASE)
    clean_prompt = re.sub(r"\s*Assistant:\s*$", "", clean_prompt, flags=re.IGNORECASE)

    best_answer = ""
    if example["answers"]:
        best_answer = example["answers"][0]["answer"]
        for ans_obj in example["answers"]:
            if ans_obj["rank"] == 1:
                best_answer = ans_obj["answer"]
                break

    return {
        "messages": [
            {"role": "user", "content": clean_prompt},
            {"role": "assistant", "content": best_answer},
        ]
    }


def transform_arena_preference(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms lmarena-ai/arena-human-preference-55k examples.
    """
    if example["winner_model_a"] or example["winner_tie"]:
        chosen_content = example["response_a"]
    else:
        chosen_content = example["response_b"]

    prompt_content = example["prompt"]
    if isinstance(prompt_content, list):
        prompt_content = "\n".join(prompt_content)

    return {
        "messages": [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": chosen_content},
        ]
    }


def transform_comparia_votes(example):
    """
    Transforms ministere-culture/comparia-votes examples.
    """
    if example["both_equal"]:
        chosen_model = example["model_a_name"]
    elif example["chosen_model_name"] is not None:
        chosen_model = example["chosen_model_name"]
    else:
        return {"messages": []}

    model_a = example["model_a_name"]
    is_a_winner = chosen_model == model_a
    model = "a" if is_a_winner else "b"

    if example.get(f"conv_incorrect_{model}") is True:
        return {"messages": []}
    if example.get(f"conv_useful_{model}") is False:
        return {"messages": []}

    selected_conversation = example.get(f"conversation_{model}")
    selected_system_prompt = example.get(f"system_prompt_{model}")

    # Handle None or empty conversation
    if not selected_conversation:
        return {"messages": []}

    messages = list(selected_conversation)

    # Filter out any None messages
    messages = [m for m in messages if m is not None and isinstance(m, dict)]

    if not messages:
        return {"messages": []}

    if selected_system_prompt and isinstance(selected_system_prompt, str):
        cleaned_prompt = selected_system_prompt.strip()
        if cleaned_prompt:
            messages.insert(0, {"role": "system", "content": cleaned_prompt})

    merged_messages = [messages[0]]
    for message in messages[1:]:
        if message.get("role") == merged_messages[-1].get("role"):
            merged_messages[-1]["content"] = (
                merged_messages[-1].get("content", "") + "\n" + message.get("content", "")
            )
        else:
            merged_messages.append(message)

    messages = [
        {"content": message.get("content", ""), "role": message.get("role", "user")}
        for message in merged_messages
    ]

    return {"messages": messages}


def transform_ultrafeedback(example):
    """
    Transforms argilla/ultrafeedback-binarized-preferences-cleaned samples.
    """
    return {"messages": example["chosen"]}


def transform_aegis_safety(example):
    """
    Transforms nvidia/Aegis-AI-Content-Safety-Dataset-2.0.
    """
    if example["prompt"] == "REDACTED":
        return {"messages": []}

    if example["response_label"] == "unsafe":
        return {"messages": []}

    if example["response_label"] == "safe":
        return {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ]
        }

    return {"messages": []}


def transform_helpsteer2(example):
    """
    Transforms nvidia/HelpSteer2.
    """
    if example["preference_strength"] >= 0:
        chosen_response = example["response_1"]
    else:
        chosen_response = example["response_2"]

    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": chosen_response},
        ]
    }


def transform_intel_orca_dpo(example):
    """
    Transforms argilla/distilabel-intel-orca-dpo-pairs.
    """
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    messages.append({"role": "user", "content": example["input"]})
    messages.append({"role": "assistant", "content": example["chosen"]})
    return {"messages": messages}


def transform_human_like_dpo(example):
    """
    Transforms HumanLLMs/Human-Like-DPO-Dataset.
    """
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]},
        ]
    }


def transform_mt_bench_judgments(example):
    """
    Transforms lmsys/mt_bench_human_judgments.
    """
    winner = example["winner"]
    if winner == "model_b":
        return {"messages": example["conversation_b"]}
    else:
        return {"messages": []}


def transform_math_preference(example):
    """
    Transforms argilla/distilabel-math-preference-dpo.
    """
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["chosen_response"]},
        ]
    }


def transform_truthy_dpo(example):
    """
    Transforms jondurbin/truthy-dpo-v0.1.
    """
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    messages.append({"role": "user", "content": example["prompt"]})
    messages.append({"role": "assistant", "content": example["chosen"]})
    return {"messages": messages}


def transform_rename_chosen_to_messages(example):
    """Simple rename of 'chosen' column to 'messages'."""
    return {"messages": example["chosen"]}


def transform_rename_conversation_to_messages(example):
    """Simple rename of 'conversation' column to 'messages'."""
    return {"messages": example["conversation"]}


def transform_identity(example):
    """Identity transform - messages column already exists."""
    return {"messages": example["messages"]}


def transform_fusion_synth(example) -> dict:
    """
    Transforms CohereLabs/fusion-synth-data-ufb examples for multilingual SFT.
    Uses the 'Fusion' completion (combined output from 5 teachers).
    Preserves 'language_code' as 'language'.
    """
    prompt = example.get("prompt", "")
    fusion = example.get("Fusion")
    if not fusion or not isinstance(fusion, dict):
        return {"messages": [], "language": ""}

    completion = fusion.get("completion", "")
    if not prompt or not completion:
        return {"messages": [], "language": ""}

    language = example.get("language_code", "")

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "language": language,
    }


# Map full language names (used by WildChat, lmsys-chat) to ISO 639-1 codes
LANGUAGE_NAME_TO_ISO = {
    "English": "en", "German": "de", "French": "fr", "Spanish": "es",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Polish": "pl",
    "Czech": "cs", "Romanian": "ro", "Greek": "el", "Ukrainian": "uk",
    "Russian": "ru", "Chinese": "zh", "Japanese": "ja", "Korean": "ko",
    "Arabic": "ar", "Turkish": "tr", "Vietnamese": "vi", "Indonesian": "id",
    "Thai": "th", "Hindi": "hi", "Swedish": "sv", "Danish": "da",
    "Finnish": "fi", "Norwegian": "no", "Hungarian": "hu", "Bulgarian": "bg",
    "Croatian": "hr", "Slovak": "sk", "Slovenian": "sl", "Estonian": "et",
    "Latvian": "lv", "Lithuanian": "lt", "Serbian": "sr", "Hebrew": "he",
    "Persian": "fa", "Catalan": "ca", "Bangla": "bn", "Tamil": "ta",
    "Urdu": "ur", "Malay": "ms", "Tagalog": "tl", "Swahili": "sw",
    "Albanian": "sq", "Icelandic": "is", "Irish": "ga", "Maltese": "mt",
    "Welsh": "cy", "Basque": "eu", "Galician": "gl", "Afrikaans": "af",
    "Georgian": "ka", "Armenian": "hy", "Azerbaijani": "az", "Kazakh": "kk",
    "Belarusian": "be", "Macedonian": "mk", "Bosnian": "bs", "Latin": "la",
}


def transform_wildchat(example) -> dict:
    """
    Transforms allenai/WildChat-1M examples for multilingual SFT.
    Filters toxic conversations, requires >=2 turns and >=50 chars.
    Preserves 'language' column.
    """
    # Filter toxic conversations
    if example.get("toxic", False):
        return {"messages": [], "language": ""}

    conversation = example.get("conversation", [])
    if not conversation or not isinstance(conversation, list):
        return {"messages": [], "language": ""}

    # Require at least 2 turns (user + assistant)
    if len(conversation) < 2:
        return {"messages": [], "language": ""}

    # Build messages
    messages = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role and content:
            messages.append({"role": role, "content": content})

    # Require minimum content length
    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars < 50:
        return {"messages": [], "language": ""}

    # Map full name to ISO code (WildChat uses "English", "German", etc.)
    raw_language = example.get("language", "")
    language = LANGUAGE_NAME_TO_ISO.get(raw_language, raw_language.lower()[:2] if raw_language else "")

    return {"messages": messages, "language": language}


def transform_lmsys_chat_multilingual(example) -> dict:
    """
    Transforms lmsys/lmsys-chat-1m examples preserving the 'language' column.
    Unlike the existing transform_rename_conversation_to_messages, this keeps language.
    Maps full language names (e.g., "English") to ISO codes (e.g., "en").
    """
    conversation = example.get("conversation", [])
    if not conversation or not isinstance(conversation, list):
        return {"messages": [], "language": ""}

    # Require at least 2 turns
    if len(conversation) < 2:
        return {"messages": [], "language": ""}

    # Require minimum content length
    total_chars = sum(len(t.get("content", "")) for t in conversation if isinstance(t, dict))
    if total_chars < 50:
        return {"messages": [], "language": ""}

    messages = []
    for turn in conversation:
        if isinstance(turn, dict) and turn.get("role") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    # Map full name to ISO code
    raw_language = example.get("language", "")
    language = LANGUAGE_NAME_TO_ISO.get(raw_language, raw_language.lower()[:2] if raw_language else "")

    return {"messages": messages, "language": language}


def transform_oasst2_dataset(ds) -> list[dict]:
    """
    Process OpenAssistant/oasst2 dataset into conversations.

    Unlike row-level transforms, this processes the entire dataset at once
    because oasst2 uses a tree structure (message_id/parent_id) that requires
    reconstructing conversation paths.

    Returns list of dicts with 'messages' and 'language' keys.
    """
    import pandas as pd

    df = ds.to_pandas()

    # Build parent->children index
    children: dict[str, list[str]] = {}
    msg_lookup: dict[str, dict] = {}
    roots: list[str] = []

    for _, row in df.iterrows():
        msg_id = row["message_id"]
        parent_id = row.get("parent_id")
        msg_lookup[msg_id] = row

        if parent_id is None or (isinstance(parent_id, float) and pd.isna(parent_id)):
            roots.append(msg_id)
        else:
            children.setdefault(parent_id, []).append(msg_id)

    def get_best_child(parent_id: str) -> str | None:
        """Get highest-ranked child message."""
        kids = children.get(parent_id, [])
        if not kids:
            return None
        # Sort by rank (lower = better), break ties by label count
        kids_with_rank = []
        for kid_id in kids:
            kid = msg_lookup[kid_id]
            rank = kid.get("rank")
            if rank is None or (isinstance(rank, float) and pd.isna(rank)):
                rank = 999
            labels = kid.get("labels", {})
            label_count = len(labels) if isinstance(labels, dict) else 0
            kids_with_rank.append((rank, -label_count, kid_id))
        kids_with_rank.sort()
        return kids_with_rank[0][2]

    conversations = []
    for root_id in roots:
        root = msg_lookup[root_id]
        # Only use prompter-initiated conversations
        if root.get("role") != "prompter":
            continue

        # Follow best-ranked path from root to leaf
        path = []
        current_id = root_id
        while current_id is not None:
            msg = msg_lookup[current_id]
            role_map = {"prompter": "user", "assistant": "assistant"}
            role = role_map.get(msg.get("role", ""), "user")
            text = msg.get("text", "")
            if text:
                path.append({"role": role, "content": text})
            current_id = get_best_child(current_id)

        if len(path) >= 2:
            # Normalize language codes: pt-BR -> pt, uk-UA -> uk, etc.
            raw_lang = root.get("lang", "")
            language = raw_lang.split("-")[0] if raw_lang else ""
            conversations.append({"messages": path, "language": language})

    return conversations


################################################################################
# Dataset mixture definitions
################################################################################

# Map transform names to functions
TRANSFORMS: Dict[str, Callable] = {
    "transform_smoltalk2": transform_smoltalk2,
    "transform_perfectblend": transform_perfectblend,
    "transform_orca_agentinstruct_1m": transform_orca_agentinstruct_1m,
    "transform_anthropic_hh_rlhf": transform_anthropic_hh_rlhf,
    "transform_stanfordnlp_shp": transform_stanfordnlp_shp,
    "transform_berkeley_nest_nectar": transform_berkeley_nest_nectar,
    "transform_arena_preference": transform_arena_preference,
    "transform_comparia_votes": transform_comparia_votes,
    "transform_ultrafeedback": transform_ultrafeedback,
    "transform_aegis_safety": transform_aegis_safety,
    "transform_helpsteer2": transform_helpsteer2,
    "transform_intel_orca_dpo": transform_intel_orca_dpo,
    "transform_human_like_dpo": transform_human_like_dpo,
    "transform_mt_bench_judgments": transform_mt_bench_judgments,
    "transform_math_preference": transform_math_preference,
    "transform_truthy_dpo": transform_truthy_dpo,
    "transform_rename_chosen_to_messages": transform_rename_chosen_to_messages,
    "transform_rename_conversation_to_messages": transform_rename_conversation_to_messages,
    "transform_identity": transform_identity,
    "transform_fusion_synth": transform_fusion_synth,
    "transform_wildchat": transform_wildchat,
    "transform_lmsys_chat_multilingual": transform_lmsys_chat_multilingual,
    # Note: oasst2 uses transform_oasst2_dataset (batch transform, not row-level)
}


# Dataset mixture definitions
DATA_MIXTURES: List[Dict[str, Any]] = [
    {
        "name": "nvidia-Nemotron-Post-Training-Dataset-v2",
        "datasets": [
            {
                "id": "nvidia/Nemotron-Post-Training-Dataset-v2",
                "config": "default",
                "split": "stem+chat+math+code+multilingual_ja+multilingual_de+multilingual_it+multilingual_es+multilingual_fr",
                "transform": "transform_identity",
            },
        ],
    },
    {
        "name": "HuggingFaceTB-smoltalk2",
        "datasets": [
            {
                "id": "HuggingFaceTB/smoltalk2",
                "config": "SFT",
                "split": "LongAlign_64k_context_lang_annotated_lang_6_no_think+Mixture_of_Thoughts_science_no_think+OpenHermes_2.5_no_think+OpenThoughts3_1.2M_no_think_no_think+hermes_function_calling_v1_no_think+smoltalk_multilingual_8languages_lang_5_no_think+smoltalk_smollm3_everyday_conversations_no_think+smoltalk_smollm3_explore_instruct_rewriting_no_think+smoltalk_smollm3_smol_magpie_ultra_no_think+smoltalk_smollm3_smol_rewrite_no_think+smoltalk_smollm3_smol_summarize_no_think+smoltalk_smollm3_systemchats_30k_no_think+table_gpt_no_think+tulu_3_sft_personas_instruction_following_no_think+xlam_traces_no_think",
                "transform": "transform_smoltalk2",
            },
            {
                "id": "HuggingFaceTB/smoltalk2",
                "config": "Preference",
                "split": "llama_3.1_tulu_3_8b_preference_mixture_no_think",
                "rename_columns": {"chosen": "messages"},
                "transform": "transform_smoltalk2",
            },
        ],
    },
    {
        "name": "mlabonne-open-perfectblend",
        "datasets": [
            {
                "id": "mlabonne/open-perfectblend",
                "config": "default",
                "split": "train",
                "transform": "transform_perfectblend",
            },
        ],
    },
    {
        "name": "microsoft-orca-agentinstruct-1M-v1",
        "datasets": [
            {
                "id": "microsoft/orca-agentinstruct-1M-v1",
                "config": "default",
                "split": "creative_content+text_modification+struct2text_flow+rc+rag+text_extraction+mcq+follow_up+analytical_reasoning+fermi+fs_cot_flow+code_+brain_teaser+text_classification+open_domain_qa",
                "transform": "transform_orca_agentinstruct_1m",
            },
        ],
    },
    {
        "name": "lmsys-lmsys-chat-1m",
        "datasets": [
            {
                "id": "lmsys/lmsys-chat-1m",
                "config": "default",
                "split": "train",
                "transform": "transform_rename_conversation_to_messages",
            },
        ],
    },
    {
        "name": "Anthropic-hh-rlhf",
        "datasets": [
            {
                "id": "Anthropic/hh-rlhf",
                "config": "default",
                "split": "train+test",
                "transform": "transform_anthropic_hh_rlhf",
            },
        ],
    },
    {
        "name": "stanfordnlp-SHP",
        "datasets": [
            {
                "id": "stanfordnlp/SHP",
                "config": "default",
                "split": "train+validation+test",
                "transform": "transform_stanfordnlp_shp",
            },
        ],
    },
    {
        "name": "berkeley-nest-Nectar",
        "datasets": [
            {
                "id": "berkeley-nest/Nectar",
                "config": "default",
                "split": "train",
                "transform": "transform_berkeley_nest_nectar",
            },
        ],
    },
    {
        "name": "lmarena-ai-arena-human-preference-55k",
        "datasets": [
            {
                "id": "lmarena-ai/arena-human-preference-55k",
                "config": "default",
                "split": "train",
                "transform": "transform_arena_preference",
            },
        ],
    },
    {
        "name": "ministere-culture-comparia-votes",
        "datasets": [
            {
                "id": "ministere-culture/comparia-votes",
                "config": "default",
                "split": "train",
                "transform": "transform_comparia_votes",
            },
        ],
    },
    {
        "name": "argilla-ultrafeedback-binarized-preferences-cleaned",
        "datasets": [
            {
                "id": "argilla/ultrafeedback-binarized-preferences-cleaned",
                "config": "default",
                "split": "train",
                "transform": "transform_ultrafeedback",
            },
        ],
    },
    {
        "name": "nvidia-Aegis-AI-Content-Safety-Dataset-2.0",
        "datasets": [
            {
                "id": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
                "config": "default",
                "split": "train+validation+test",
                "transform": "transform_aegis_safety",
            },
        ],
    },
    {
        "name": "nvidia-HelpSteer2",
        "datasets": [
            {
                "id": "nvidia/HelpSteer2",
                "config": "default",
                "data_dir": "preference",
                "split": "train",
                "transform": "transform_helpsteer2",
            },
        ],
    },
    {
        "name": "argilla-distilabel-intel-orca-dpo-pairs",
        "datasets": [
            {
                "id": "argilla/distilabel-intel-orca-dpo-pairs",
                "config": "default",
                "split": "train",
                "transform": "transform_intel_orca_dpo",
            },
        ],
    },
    {
        "name": "HumanLLMs-Human-Like-DPO-Dataset",
        "datasets": [
            {
                "id": "HumanLLMs/Human-Like-DPO-Dataset",
                "config": "default",
                "split": "train",
                "transform": "transform_human_like_dpo",
            },
        ],
    },
    {
        "name": "argilla-distilabel-capybara-dpo-7k-binarized",
        "datasets": [
            {
                "id": "argilla/distilabel-capybara-dpo-7k-binarized",
                "config": "default",
                "split": "train",
                "transform": "transform_rename_chosen_to_messages",
            },
        ],
    },
    {
        "name": "lmsys-mt_bench_human_judgments",
        "datasets": [
            {
                "id": "lmsys/mt_bench_human_judgments",
                "config": "default",
                "split": "human",
                "transform": "transform_mt_bench_judgments",
            },
        ],
    },
    {
        "name": "argilla-distilabel-math-preference-dpo",
        "datasets": [
            {
                "id": "argilla/distilabel-math-preference-dpo",
                "config": "default",
                "split": "train",
                "transform": "transform_math_preference",
            },
        ],
    },
    {
        "name": "jondurbin-truthy-dpo-v0.1",
        "datasets": [
            {
                "id": "jondurbin/truthy-dpo-v0.1",
                "config": "default",
                "split": "train",
                "transform": "transform_truthy_dpo",
            },
        ],
    },
    # --- Multilingual datasets (for EU language fine-tuning) ---
    {
        "name": "CohereLabs-fusion-synth-data-ufb",
        "datasets": [
            {
                "id": "CohereLabs/fusion-synth-data-ufb",
                "config": "default",
                "split": "train",
                "transform": "transform_fusion_synth",
            },
        ],
        "keep_columns": ["messages", "language"],
    },
    {
        "name": "allenai-WildChat-1M",
        "datasets": [
            {
                "id": "allenai/WildChat-1M",
                "config": "default",
                "split": "train",
                "transform": "transform_wildchat",
            },
        ],
        "keep_columns": ["messages", "language"],
    },
    {
        "name": "lmsys-lmsys-chat-1m-multilingual",
        "datasets": [
            {
                "id": "lmsys/lmsys-chat-1m",
                "config": "default",
                "split": "train",
                "transform": "transform_lmsys_chat_multilingual",
            },
        ],
        "keep_columns": ["messages", "language"],
    },
    {
        "name": "OpenAssistant-oasst2",
        "custom_process": "oasst2",
        "datasets": [
            {
                "id": "OpenAssistant/oasst2",
                "config": "default",
                "split": "train+validation",
            },
        ],
        "keep_columns": ["messages", "language"],
    },
]


################################################################################
# Processing functions
################################################################################


def load_and_transform_dataset(
    dataset_id: str,
    config: str,
    split: str,
    transform_name: str,
    data_dir: Optional[str] = None,
    rename_columns: Optional[Dict[str, str]] = None,
    keep_columns: Optional[List[str]] = None,
    num_proc: int = 8,
) -> Dataset:
    """Load a dataset and apply transformation."""
    print(f"  Loading {dataset_id} (config={config}, split={split})...")

    load_kwargs = {
        "path": dataset_id,
        "split": split,
        "trust_remote_code": True,
    }
    if config and config != "default":
        load_kwargs["name"] = config
    if data_dir:
        load_kwargs["data_dir"] = data_dir

    try:
        ds = load_dataset(**load_kwargs)
    except (TypeError, ValueError) as e:
        if "trust_remote_code" in str(e):
            # Newer HF datasets versions don't support trust_remote_code
            load_kwargs.pop("trust_remote_code", None)
            ds = load_dataset(**load_kwargs)
        else:
            print(f"  ERROR loading dataset: {e}")
            return None
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        return None

    print(f"  Loaded {len(ds)} examples")

    # Rename columns if specified (before transform)
    if rename_columns:
        for old_name, new_name in rename_columns.items():
            if old_name in ds.column_names:
                ds = ds.rename_column(old_name, new_name)
                print(f"  Renamed column: {old_name} -> {new_name}")

    # Apply transform
    transform_fn = TRANSFORMS.get(transform_name)
    if transform_fn is None:
        raise ValueError(f"Unknown transform: {transform_name}")

    # Determine which columns to keep after transform
    if keep_columns is None:
        keep_columns = ["messages"]
    columns_to_remove = [c for c in ds.column_names if c not in keep_columns]

    print(f"  Applying transform: {transform_name}...")
    # Use num_proc=1 to avoid Arrow schema inference issues when multiprocessing
    # encounters empty messages [] in some processes and valid messages in others
    ds = ds.map(
        transform_fn,
        num_proc=1,  # Single process avoids schema conflicts across workers
        remove_columns=columns_to_remove,
        desc=f"Transforming {dataset_id}",
    )

    # Filter out empty messages
    original_len = len(ds)
    ds = ds.filter(lambda x: len(x["messages"]) > 0, num_proc=num_proc)
    filtered_len = len(ds)
    if filtered_len < original_len:
        print(f"  Filtered {original_len - filtered_len} empty examples")

    print(f"  Final: {len(ds)} examples")
    return ds


def process_mixture(
    mixture: Dict[str, Any],
    output_dir: Path,
    num_proc: int = 8,
    force: bool = False,
) -> Optional[Path]:
    """Process a single dataset mixture."""
    mixture_name = mixture["name"]
    output_path = output_dir / f"{mixture_name}.parquet"
    metadata_path = output_dir / f"{mixture_name}_metadata.json"

    # Skip if already exists with valid hash (unless force is set)
    if output_path.exists() and not force:
        # Check if metadata exists and hash is valid
        hash_valid = False
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                expected_hash = metadata.get("sha256")
                if expected_hash:
                    print(f"\n  Verifying hash for {mixture_name}...", end=" ")
                    hash_valid = verify_file_hash(output_path, expected_hash)
                    if hash_valid:
                        print("OK")
                    else:
                        print("MISMATCH - will reprocess")
            except (json.JSONDecodeError, IOError):
                pass

        if hash_valid:
            print(f"\n{'='*60}")
            print(f"Skipping mixture: {mixture_name} (exists with valid hash)")
            print(f"  File: {output_path}")
            print(f"  Use --force to reprocess")
            print(f"{'='*60}")
            return output_path
        elif metadata_path.exists():
            # File exists but hash invalid or missing - reprocess
            print(f"  Hash verification failed, reprocessing {mixture_name}...")

    print(f"\n{'='*60}")
    print(f"Processing mixture: {mixture_name}")
    print(f"{'='*60}")

    keep_columns = mixture.get("keep_columns")

    # Handle custom processing (e.g., oasst2 tree reconstruction)
    if mixture.get("custom_process") == "oasst2":
        ds_config = mixture["datasets"][0]
        print(f"  Loading {ds_config['id']} for tree reconstruction...")
        load_kwargs = {
            "path": ds_config["id"],
            "split": ds_config["split"],
            "trust_remote_code": True,
        }
        config_name = ds_config.get("config", "default")
        if config_name and config_name != "default":
            load_kwargs["name"] = config_name

        try:
            raw_ds = load_dataset(**load_kwargs)
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            return None

        print(f"  Loaded {len(raw_ds)} messages, reconstructing conversation trees...")
        conversations = transform_oasst2_dataset(raw_ds)
        print(f"  Reconstructed {len(conversations)} conversations")

        if not conversations:
            print(f"  No conversations extracted for {mixture_name}")
            return None

        combined = Dataset.from_list(conversations)
    else:
        all_datasets = []

        for ds_config in mixture["datasets"]:
            ds = load_and_transform_dataset(
                dataset_id=ds_config["id"],
                config=ds_config.get("config", "default"),
                split=ds_config["split"],
                transform_name=ds_config["transform"],
                data_dir=ds_config.get("data_dir"),
                rename_columns=ds_config.get("rename_columns"),
                keep_columns=keep_columns,
                num_proc=num_proc,
            )
            if ds is not None and len(ds) > 0:
                all_datasets.append(ds)

        if not all_datasets:
            print(f"  No data loaded for mixture {mixture_name}")
            return None

        # Concatenate all datasets
        if len(all_datasets) > 1:
            print(f"  Concatenating {len(all_datasets)} datasets...")
            combined = concatenate_datasets(all_datasets)
        else:
            combined = all_datasets[0]

    print(f"  Total examples: {len(combined)}")

    # Save to parquet (compatible with convert_sft_data_for_olmocore.py)
    print(f"  Saving to {output_path}...")
    combined.to_parquet(str(output_path))

    # Compute hash for integrity verification
    print(f"  Computing SHA256 hash...")
    file_hash = compute_file_hash(output_path)

    # Save metadata with hash
    metadata = {
        "mixture_name": mixture_name,
        "num_examples": len(combined),
        "source_datasets": [ds["id"] for ds in mixture["datasets"]],
        "parquet_file": str(output_path),
        "sha256": file_hash,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved successfully! (sha256: {file_hash[:16]}...)")
    return output_path


################################################################################
# Main
################################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess SFT datasets for OLMo-core training"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets_mixture_sft_preprocessed"),
        help="Output directory for preprocessed datasets",
    )
    parser.add_argument(
        "--mixtures",
        nargs="+",
        default=None,
        help="Specific mixture names to process (default: all)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes for dataset operations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing",
    )
    parser.add_argument(
        "--list-mixtures",
        action="store_true",
        help="List all available mixture names and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output file already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # List mixtures and exit
    if args.list_mixtures:
        print("Available mixtures:")
        for mixture in DATA_MIXTURES:
            print(f"  - {mixture['name']}")
        return

    # Filter mixtures if specified
    if args.mixtures:
        mixtures_to_process = [
            m for m in DATA_MIXTURES
            if any(pattern in m["name"] for pattern in args.mixtures)
        ]
    else:
        mixtures_to_process = DATA_MIXTURES

    if not mixtures_to_process:
        print("No mixtures match the specified patterns!")
        return

    print(f"Will process {len(mixtures_to_process)} mixture(s):")
    for m in mixtures_to_process:
        print(f"  - {m['name']}")

    if args.dry_run:
        print("\n[DRY RUN] Would process the above mixtures")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each mixture
    processed_names = []
    skipped_names = []
    for mixture in mixtures_to_process:
        output_path = args.output_dir / f"{mixture['name']}.parquet"
        already_exists = output_path.exists() and not args.force

        result = process_mixture(
            mixture=mixture,
            output_dir=args.output_dir,
            num_proc=args.num_proc,
            force=args.force,
        )
        if result:
            if already_exists:
                skipped_names.append(mixture["name"])
            else:
                processed_names.append(mixture["name"])

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"Processed {len(processed_names)} mixture(s), skipped {len(skipped_names)} (already exist)")
    print(f"Output directory: {args.output_dir}")
    print("")
    if processed_names:
        print("Newly processed datasets:")
        for name in processed_names:
            print(f"  - {args.output_dir}/{name}.parquet")
    if skipped_names:
        print("Skipped (already exist, use --force to reprocess):")
        for name in skipped_names:
            print(f"  - {args.output_dir}/{name}.parquet")
    print("")
    print("Next steps:")
    print("  For baseline reproduction (100% English):")
    print("    sbatch oellm/pipelines/tokenization/tokenize_baseline.sh")
    print("")
    print("  For EU24 experiments (with language mixing):")
    print("    1. Translate: sbatch oellm/pipelines/translation/translate_slurm.sh")
    print("    2. Tokenize:  CONFIG=mixture_eu24_90en_10eu.yaml sbatch oellm/pipelines/tokenization/tokenize_eu24.sh")


if __name__ == "__main__":
    main()
