"""
Synthetic e-commerce product data generator.

Generates product text descriptions and placeholder images for each leaf category.
Products are split into anchor (70%) and test (30%) sets.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --- Product description templates ---

BRAND_POOL = {
    "服装鞋帽": ["Nike", "Adidas", "优衣库", "ZARA", "H&M", "李宁", "安踏", "海澜之家", "太平鸟", "GXG"],
    "数码电器": ["苹果", "华为", "小米", "三星", "联想", "戴尔", "索尼", "佳能", "大疆", "海尔"],
    "食品饮料": ["三只松鼠", "良品铺子", "百草味", "农夫山泉", "蒙牛", "伊利", "茅台", "青岛啤酒", "雀巢", "星巴克"],
    "美妆个护": ["兰蔻", "雅诗兰黛", "欧莱雅", "SK-II", "资生堂", "完美日记", "花西子", "珀莱雅", "薇诺娜", "百雀羚"],
    "家居家装": ["宜家", "顾家", "全友", "林氏木业", "源氏木语", "水星家纺", "罗莱", "九牧", "箭牌", "松下"],
    "运动户外": ["迪卡侬", "探路者", "北面", "始祖鸟", "哥伦比亚", "李宁", "安踏", "特步", "匹克", "361度"],
    "母婴用品": ["好孩子", "贝亲", "花王", "帮宝适", "惠氏", "美赞臣", "巴拉巴拉", "安奈儿", "英氏", "嫚熙"],
    "图书文具": ["人民文学", "中信出版", "晨光", "得力", "真彩", "斑马", "百乐", "辉柏嘉", "马利", "步步高"],
    "汽车用品": ["3M", "龟牌", "固特异", "博世", "米其林", "壳牌", "嘉实多", "飞利浦", "70迈", "盯盯拍"],
    "宠物用品": ["皇家", "渴望", "冠能", "伯纳天纯", "网易严选", "疯狂小狗", "卫仕", "麦富迪", "比瑞吉", "耐威克"],
    "珠宝饰品": ["周大福", "周生生", "老凤祥", "六福", "潘多拉", "施华洛世奇", "蒂芙尼", "卡地亚", "宝格丽", "周大生"],
}

MATERIAL_POOL = ["优质", "高端", "经典", "时尚", "简约", "轻奢", "百搭", "新款", "热卖", "限定"]
FEATURE_POOL = ["舒适", "耐用", "透气", "防水", "轻便", "保暖", "柔软", "坚固", "环保", "高性能"]
OCCASION_POOL = ["日常", "通勤", "聚会", "运动", "旅行", "居家", "商务", "户外", "节日", "送礼"]


def generate_product_description(l1: str, l2: str, leaf: str, idx: int) -> dict:
    """Generate a synthetic product with text description."""
    brands = BRAND_POOL.get(l1, ["品牌A", "品牌B", "品牌C"])
    brand = random.choice(brands)
    material = random.choice(MATERIAL_POOL)
    feature1, feature2 = random.sample(FEATURE_POOL, 2)
    occasion = random.choice(OCCASION_POOL)

    # Title: brand + adjectives + leaf category name
    title = f"{brand} {material}{leaf}"

    # Description: more detailed
    templates = [
        f"{brand}品牌{material}{leaf}，{feature1}{feature2}，适合{occasion}使用。精选优质材料，品质保证。",
        f"这款{leaf}来自{brand}，采用{material}设计，具有{feature1}和{feature2}的特点，{occasion}必备好物。",
        f"{brand}{material}款{leaf}，主打{feature1}{feature2}，专为{occasion}场景打造，性价比之选。",
        f"全新{brand}{leaf}，{material}风格，{feature1}又{feature2}，无论{occasion}都能轻松驾驭。",
        f"{occasion}推荐：{brand}{material}{leaf}，{feature1}体验，{feature2}品质，值得入手。",
    ]
    description = random.choice(templates)

    price = round(random.uniform(9.9, 9999.0), 2)

    return {
        "product_id": f"{leaf}_{idx:04d}",
        "title": title,
        "description": description,
        "price": price,
        "brand": brand,
        "category_l1": l1,
        "category_l2": l2,
        "category_leaf": leaf,
    }


def generate_product_image(product_id: str, leaf: str, image_dir: Path, size: int = 224) -> str:
    """
    Generate a synthetic product image.

    Uses colored rectangles with category-based color mapping to create
    visually distinguishable images per category. This ensures the model
    can learn category-image associations.
    """
    # Deterministic color based on leaf category name
    rng = random.Random(hash(leaf))
    base_r = rng.randint(30, 225)
    base_g = rng.randint(30, 225)
    base_b = rng.randint(30, 225)

    # Add per-product variation
    r = min(255, max(0, base_r + random.randint(-30, 30)))
    g = min(255, max(0, base_g + random.randint(-30, 30)))
    b = min(255, max(0, base_b + random.randint(-30, 30)))

    # Create image with colored background and some pattern
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = [r, g, b]

    # Add a simple geometric pattern for visual variety
    pattern_type = random.randint(0, 3)
    pr, pg, pb = min(255, r + 40), min(255, g + 40), min(255, b + 40)

    if pattern_type == 0:  # Horizontal stripe
        y1 = random.randint(size // 4, size // 2)
        y2 = y1 + random.randint(20, 60)
        img[y1:min(y2, size), :] = [pr, pg, pb]
    elif pattern_type == 1:  # Vertical stripe
        x1 = random.randint(size // 4, size // 2)
        x2 = x1 + random.randint(20, 60)
        img[:, x1:min(x2, size)] = [pr, pg, pb]
    elif pattern_type == 2:  # Center rectangle
        margin = size // 4
        img[margin:size - margin, margin:size - margin] = [pr, pg, pb]
    else:  # Corner blocks
        q = size // 2
        img[:q, :q] = [pr, pg, pb]

    pil_img = Image.fromarray(img)
    image_path = image_dir / f"{product_id}.jpg"
    pil_img.save(str(image_path), quality=85)

    return str(image_path)


def load_categories(categories_file: str) -> list:
    """Load category taxonomy and return flat list of (l1, l2, leaf) tuples."""
    with open(categories_file, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    categories = []
    for l1, l2_dict in taxonomy.items():
        for l2, leaves in l2_dict.items():
            for leaf in leaves:
                categories.append((l1, l2, leaf))
    return categories


def main():
    random.seed(42)
    np.random.seed(42)

    # Paths
    data_dir = PROJECT_ROOT / "data"
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    categories_file = data_dir / "categories.json"
    categories = load_categories(str(categories_file))

    print(f"Category taxonomy loaded:")
    l1_set = set(c[0] for c in categories)
    l2_set = set((c[0], c[1]) for c in categories)
    print(f"  L1 categories: {len(l1_set)}")
    print(f"  L2 categories: {len(l2_set)}")
    print(f"  Leaf categories: {len(categories)}")

    # Generate products
    num_per_leaf = 40
    anchor_ratio = 0.7
    all_products = []

    print(f"\nGenerating {num_per_leaf} products per leaf category...")
    for l1, l2, leaf in categories:
        for idx in range(num_per_leaf):
            product = generate_product_description(l1, l2, leaf, idx)
            image_path = generate_product_image(product["product_id"], leaf, image_dir)
            product["image_path"] = image_path
            all_products.append(product)

    df = pd.DataFrame(all_products)
    print(f"Total products generated: {len(df)}")

    # Split into anchor and test
    anchor_dfs = []
    test_dfs = []
    for (l1, l2, leaf), group in df.groupby(["category_l1", "category_l2", "category_leaf"]):
        n_anchor = int(len(group) * anchor_ratio)
        shuffled = group.sample(frac=1, random_state=42)
        anchor_dfs.append(shuffled.iloc[:n_anchor])
        test_dfs.append(shuffled.iloc[n_anchor:])

    anchor_df = pd.concat(anchor_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Save
    df.to_csv(data_dir / "products.csv", index=False, encoding="utf-8-sig")
    anchor_df.to_csv(data_dir / "anchor_products.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(data_dir / "test_products.csv", index=False, encoding="utf-8-sig")

    print(f"\nData saved:")
    print(f"  All products: {len(df)} → data/products.csv")
    print(f"  Anchor products: {len(anchor_df)} → data/anchor_products.csv")
    print(f"  Test products: {len(test_df)} → data/test_products.csv")
    print(f"  Images: data/images/ ({len(df)} files)")

    # Summary stats
    print(f"\nSummary:")
    print(f"  L1 categories: {df['category_l1'].nunique()}")
    print(f"  L2 categories: {df['category_l2'].nunique()}")
    print(f"  Leaf categories: {df['category_leaf'].nunique()}")
    print(f"  Anchor/Test ratio: {len(anchor_df)}/{len(test_df)} "
          f"({len(anchor_df)/len(df):.1%}/{len(test_df)/len(df):.1%})")


if __name__ == "__main__":
    main()
