# 部署指南

## 服务器要求

- GPU: A100 40GB（推荐）/ RTX 3090 24GB（最低）
- 内存: 64GB
- 硬盘: 100GB SSD
- 镜像: PyTorch 2.1 + CUDA 12.1（AutoDL 可直接选）

## 一键部署

```bash
# 1. 克隆代码
git clone <your-repo-url> product-category-recognition
cd product-category-recognition

# 2. 安装环境 + 下载模型（约10-15分钟）
bash setup.sh

# 3. 激活环境
conda activate blip2-product

# 4. 一键跑通全流程（数据生成 → 离线embedding → 评测）
make all
```

完成后查看 `results/metrics.json` 获取指标。

## 分步运行（调试用）

```bash
# Step 1: 生成合成数据（~10秒）
make data

# Step 2: 离线阶段 - 生成embedding + 建索引（A100约20min）
make offline

# Step 3: 评测（A100约10min）
make evaluate

# 可选: 单商品演示
make demo

# 可选: 单条推理
python scripts/run_online.py \
  --title "Nike Air Max 运动鞋" \
  --description "舒适透气的跑步鞋" \
  --image data/images/运动鞋_0001.jpg
```

## 调参指南

编辑 `config/default.yaml`：

```yaml
online:
  recall:
    top_n: 50        # 召回数量，增大可提高召回率但降低精度
  ranking:
    top_n: 10        # 排序保留数量
  rerank:
    recall_weight: 0.45  # 召回分数权重
    rank_weight: 0.55    # 排序分数权重，两者之和=1
```

调参后只需重新运行评测（无需重跑离线）：
```bash
make evaluate
```

## 输出文件

```
results/
├── metrics.json                    # TOP-K 准确率
├── classification_report_l1.txt    # 一级类目分类报告
├── classification_report_l2.txt    # 二级类目分类报告
├── confusion_matrix_l1.png         # 一级类目混淆矩阵
└── confusion_matrix_l2.png         # 二级类目混淆矩阵
```

## 清理重跑

```bash
make clean   # 清除所有生成文件
make all     # 重新跑全流程
```

## 常见问题

### CUDA out of memory
降低 `config/default.yaml` 中的 `batch_size`：
```yaml
offline:
  batch_size: 16  # 从32降到16或8
```

### 模型下载慢
使用HuggingFace镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
bash setup.sh
```

### AutoDL 特殊配置
AutoDL 自带conda，可跳过conda创建步骤：
```bash
pip install -r requirements.txt
python -c "from transformers import Blip2Processor; Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')"
make all
```
