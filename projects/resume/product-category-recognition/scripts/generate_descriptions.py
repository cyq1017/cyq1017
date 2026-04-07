# scripts/generate_descriptions.py
"""Generate rich category descriptions via LLM API."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Allow tests to run without openai installed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_description(client, category_name: str,
                         model: str = "M2.7") -> str:
    """Generate a rich description for a single category."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"给出'{category_name}'商品类目的详细描述（50-100字），"
                       f"包括典型品类词、常见品牌、商品特征。只输出描述，不要标题。"
        }],
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def generate_with_cache(client, category_name: str,
                        cache: dict, model: str = "M2.7") -> str:
    """Generate description with cache, fallback on error."""
    if category_name in cache:
        return cache[category_name]
    try:
        desc = generate_description(client, category_name, model=model)
    except Exception as e:
        print(f"  API error for {category_name}: {e}, using fallback")
        desc = f"{category_name}，电商商品类目"
    cache[category_name] = desc
    return desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories-file", required=True,
                        help="Path to v2_train.csv or categories list")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data/category_descriptions.json"))
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="https://api.minimax.io/v1")
    parser.add_argument("--model", default="M2.7")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    output_path = Path(args.output)

    # Load existing cache
    cache = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # Get unique categories
    import pandas as pd
    df = pd.read_csv(args.categories_file)
    categories = sorted(df["category_leaf"].unique()) if "category_leaf" in df.columns \
        else sorted(df["class"].astype(str).unique())

    print(f"Generating descriptions for {len(categories)} categories "
          f"({len(cache)} cached)")

    for i, cat in enumerate(categories):
        generate_with_cache(client, str(cat), cache, model=args.model)
        if (i + 1) % 50 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f"  Progress: {i+1}/{len(categories)}")

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cache)} descriptions to {output_path}")


if __name__ == "__main__":
    main()
