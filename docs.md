# 测试AI模型安全性评估API

## robustness

### GET `robustness/`

返回鲁棒性评估起始页面

### POST `robustness/`

返回评估结果

### Query Parameters

| parameters | type   | descriptions      |
| ---------- | ------ | ----------------- |
| `model`    | string | 所选的鲁棒性增强方法        |
| `label`    | string | 上传的图像标签（ImageNet） |
| `img`      | file   | 上传的图像             |

### Return Field

| parameters | type | descriptions   |
| ---------- | ---- | -------------- |
| `pth1`     | file | 普通模型对抗攻击结果图像   |
| `pth2`     | file | 鲁班增强模型对抗攻击结果图像 |

## fairness

### GET `fairness/`

返回去偏评估起始页面

### POST `fairness/`

返回评估结果

以预测男女性的工资收入为例

### Return Field

| parameters  | type | descriptions            |
| ----------- | ---- | ----------------------- |
| `raw_path`  | file | 普通模型结果                  |
| `demo_path` | file | 去偏方法Demographicparity   |
| `adv_path`  | file | 去偏方法Adversarialfairness |