# MedLLM

<div align="center">
  

## 概述

MedLLM 是一个专为医疗健康对话场景而打造的领域大模型，它可以满足您的各种医疗保健需求，包括疾病问诊和治疗方案咨询等，为您提供高质量的健康支持服务。MedLLM 有效地对齐了医疗场景下的人类偏好，弥合了通用语言模型输出与真实世界医疗对话之间的差距。

## 部署

当前版本的 MedLLM 是基于[Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)训练得到的。您可以直接从 [BaiduDisk](https://pan.baidu.com/s/19ryE1OFRfkeQ_ZRRIsL7yA?pwd=3wqb) 上下载我们的模型权重。

本项目流畅运行需要先下载 [MeloTTS](https://github.com/myshell-ai/MeloTTS) 到到项目目录并按照提示配置好相应环境，接着在环境变量里面设置成功 [DeepSeek API](https://platform.deepseek.com/usage)。

首先，您需要安装项目的依赖环境（建议Pytorch安装GPU版本）。
```shell
pip install -r requirements.txt
```

### 利用预训练模型进行推理
```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from transformers.generation.utils import GenerationConfig
>>> tokenizer = AutoTokenizer.from_pretrained("Baichuan2-7B-MedLLM-Merged", use_fast=False, trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("Baichuan2-7B-MedLLM-Merged", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
>>> model.generation_config = GenerationConfig.from_pretrained("Baichuan2-7B-MedLLM-Merged")
>>> messages = []
>>> messages.append({"role": "user", "content": "我感觉自己颈椎非常不舒服，每天睡醒都会头痛"})
>>> response = model.chat(tokenizer, messages)
>>> print(response)
```

### 推荐：运行网页版Demo
```shell
streamlit run voice_doctor_webui.py 
```

### 运行离线转译脚本
```shell
python offline_transcribe.py 
```
<br>

## 对模型进行微调
您可以使用与我们的数据集结构相同的数据对我们的模型进行微调。我们的训练代码在 [Firefly](https://github.com/yangjianxin1/Firefly) 的基础上进行了修改，使用了不同的数据结构和对话格式。这里我们只提供全参数微调的代码：
```shell
deepspeed --num_gpus={num_gpus} ./train/train.py 
```
请您在开始进行模型训练前检查 `sft.json` 中的设置。

## 模型评测
<!-- We compare our model with three general-purpose LLMs and two conversational Chinese medical domain LLMs. Specifically, these are GPT-3.5 and GPT-4 from OpenAI, the aligned conversational version of our backbone model Baichuan-13B-Base, Baichuan-13B-Chat, and the open-source Chinese conversational medical model HuatuoGPT-13B (trained from Ziya-Llama-13B) and BianQue-2. Our evaluation approach encompasses two key dimensions: an assessment of conversational aptitude using GPT-4 as a reference judge, and a comprehensive benchmark evaluation. -->

我们从两个角度评估了模型的性能，包括在单轮QA问题中提供准确答案的能力以及在多轮对话中完成系统性问诊、解决咨询需求的能力。

* 在单轮对话评测中，我们构建了一个基准测试数据集，其中包含从两个公开医疗数据集中收集的多项选择题，并评估模型回答的准确性。
* 对于多轮对话评测，我们首先构建了一些高质量的诊疗对话案例，然后让 GPT-3.5 扮演这些案例中的患者角色，并与扮演医生角色的模型进行对话。我们利用 GPT-4 来评估整段每段对话的**主动性**、**准确性**, **帮助性**和**语言质量**。

您可以在 `eval/` 目录下查看测试数据集、各个模型生成的对话结果以及 GPT-4 提供的打分结果。<br>

## 声明
由于语言模型固有的局限性，我们无法保证 MedLLM 模型所生成的信息的准确性或可靠性。该模型仅为个人和学术团体的研究和测试而设计。我们敦促用户以批判性的眼光对模型输出的任何信息或医疗建议进行评估，并且强烈建议不要盲目信任此类信息结果。我们不对因使用该模型所引发的任何问题、风险或不良后果承担责任。

##  致谢
复旦大学的DISC-MedLLM提供了本项目所用的微调数据集和代码思路，引用链接如下：
```angular2
@misc{bao2023discmedllm,
      title={DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation}, 
      author={Zhijie Bao and Wei Chen and Shengze Xiao and Kuang Ren and Jiaao Wu and Cheng Zhong and Jiajie Peng and Xuanjing Huang and Zhongyu Wei},
      year={2023},
      eprint={2308.14346},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```




