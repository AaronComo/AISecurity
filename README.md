# AI模型鲁棒性和公平性评估

## 鲁棒性评估

1. 使用方法：

    - 可选模型: resnet50、cspresnet50、efficientnet_b0、xception、densenet121、fbnetc_100、mobilenetv2_100、resnext101_32x8d
    - 上传图片要求
        - png 格式
        - 文件名为 ImageNet 数据集中的 class 值, 如 `n02085620`

2. 流程

    - 上传图片后, 将使用 PGD 攻击方法产生一个对抗性样本~~, 默认目标标签如下,~~ 可在 eval.py 中更改 `target_label` 来适配自己的需求, 生成的对抗性样本为 `./static/robustness/adversarial.png`

        ~~~json
          "10": [
            "n01530575",
            "brambling"
          ]
        ~~~

    - 系统使用对抗性样本在目录 `./static/robustness` 下生成 `target_{target_label}.png`, 这是对普通模型的攻击效果, 同时跳转到新页面展示原始样本和攻击效果

    - 点击 `开始识别` , 系统将使用对抗性样本攻击选择的鲁棒性模型, 生成 `robust_{name}.png`, 这是对鲁棒模型name的攻击效果

3. API

    1. 上传图片

        - 按钮: 上传图片

        - 调用函数: `uploadImg()`

        - 作用: 生成对抗性样本、生成攻击普通模型的结果

        - 调用参数

            ~~~python
            QueryDict = {
                "model": 待测模型名称
                "img": 上传图像
            }
            ~~~
    
        - 返回值

            ~~~python
            cont = {
                "orl_path": 原始样本路径
                "adversarial": 对抗性样本路径
                "adv_path": 对普通模型攻击效果图路径
                "model": 待测试鲁棒性的模型名称
                "true": 正确标签
                "pre": 非鲁棒模型预测标签
            }
            ~~~
    
    2. 鲁棒性测试
    
        - 按钮: 开始识别
    
        - 调用函数: `testRobustness()`
    
        - 作用: 生成攻击鲁棒性模型的结果
    
        - 调用参数
    
            ~~~python
            request = {
                "adversarial": 对抗性样本路径
                "model": 待测试鲁棒性的模型名称
            }
            ~~~
        
        - 返回值
        
            ~~~python
            cont = {
                "adversarial": 对抗性样本路径
                "robust_path": 对鲁棒性模型攻击效果路径
                "model": 待测试鲁棒性的模型名称
                "pre": 鲁棒模型预测标签
            }
            
            # 以上路径均为网络路径
            ~~~
            
        



## 公平性评估

1. 使用方法: 点击开始按钮

2. 功能: 对存在性别歧视的男女工资预测模型进行去偏, 在 `static/fairness/` 下生成三张结果图

3. API

    - 按钮: 开始

    - 调用函数: `start()`

    - 调用参数: 无

    - 返回值
    
        ~~~python
        cont = {
            "raw_path": 为去偏的模型结果
            "demo_path": 应用偏见去除方法Demographicparity的结果
            "adv_path": 应用对抗性偏见去除方法Adversarialfairness的结果
        }
        
        # 以上路径均为网络路径
        ~~~

        
    
    



------



## 以下为修改日志

### 公平性评估(去偏)

1. 新增

    - `fairnes/` 中为去偏页面后端文件
    - `templates/fairness/` 包含所用到的三个页面的 HTML 文件
    - 在 settings.py 中加入引用静态目录语句(应该是项目之前留下的bug)

2. 更改

    - 更改 settings.py、 urls.py 注册新的 app
    - 将去偏的生成目录改为 `./static/fairness` 以使网页能够正确加载图像
    - 为 fairness.py 中的函数增加参数, 将生成目录作为参数传入
    - 重写API

3. 问题

    1. ~~`fairness.py` 中仅更改了函数参数, 但是存在如下语句~~

        ~~~python
        from .models import PredictorModel
        ~~~

        ~~在我的本地会报错, 提示找不到 `PredictorModel`, 代码中没有提示此类的用处, 看README感觉写去偏代码的学长跑通了, 我就没管这里~~
    
    2. ~~给到我的前端代码无法显示图片~~



### 鲁棒性评估

1. 更改
    - 函数重命名
        - `robustness` &rightarrow; `adv_attack`
        - `robust` &rightarrow; `test_robust`
    - 重写所有 API
    - 重构 robustness.py
    - 修改对抗性样本目标为随机数



### 整体

1. 前后端解耦合
2. 调整所有接口, 返回 JSON 数据
3. 增加安全性检查
