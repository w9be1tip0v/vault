---
title: "OpenAI GPT OSS Model Card"
source: "Clippings/oai_gpt-oss_model_card.pdf"
author: "OpenAI"
published: "2025-08-05"
created: 2025-01-27
description: "Model card documentation for OpenAI's GPT OSS (Open Source Software) models - gpt-oss-120b and gpt-oss-20b, providing comprehensive information about model architecture, training data, capabilities, limitations, and responsible AI considerations."
tags:
  - type/literature
  - theme/research
  - theme/learning
  - source/openai
  - keyword/gpt
  - keyword/language-model
  - keyword/ai
  - keyword/nlp
  - keyword/model-card
  - keyword/responsible-ai
  - keyword/open-source
  - keyword/machine-learning
  - keyword/mixture-of-experts
  - keyword/reasoning
  - keyword/tool-use
---

# gpt-oss-120b & gpt-oss-20b Model Card

OpenAI  
August 5, 2025

## 1 Contents

1 Introduction 3  
2 Model architecture, data, training and evaluations 3  
2.1 Quantization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4  
2.2 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4  
2.3 Tokenizer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5  
2.4 Pretraining . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5  
2.5 Post-Training for Reasoning and Tool Use . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6  
2.5.1 Harmony Chat Format . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6  
2.5.2 Variable Effort Reasoning Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7  
2.5.3 Agentic Tool Use . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7  
2.6 Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7  
2.6.1 Reasoning, Factuality and Tool Use . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8  
2.6.2 Health Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8  
2.6.3 Multilingual Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9  
2.6.4 Full Evaluations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10  
3 Safety testing and mitigation approach 10  
4 Default Safety Performance: Observed Challenges and Evaluations 11  
4.1 Disallowed Content . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11  
4.2 Jailbreaks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13  
4.3 Instruction Hierarchy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13  
4.4 Hallucinated chains of thought . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15  
4.5 Hallucinations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16  
4.6 Fairness and Bias . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16  
5 Preparedness Framework 16  
5.1 Adversarial Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17  
15.1.1 External Safety expert feedback on adversarial training methodology . . . 17  
5.2 Capability findings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18  
5.2.1 Biological and Chemical - Adversarially Fine-tuned . . . . . . . . . . . . . . . . . 18  
5.2.1.1 Long-form Biological Risk Questions . . . . . . . . . . . . . . . . . . . . 19  
5.2.1.2 Multimodal Troubleshooting Virology . . . . . . . . . . . . . . . . . . . 20  
5.2.1.3 ProtocolQA Open-Ended . . . . . . . . . . . . . . . . . . . . . . . . . . 20  
5.2.1.4 Tacit Knowledge and Troubleshooting . . . . . . . . . . . . . . . . . . . 21  
5.2.1.5 TroubleshootingBench . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21  
5.2.1.6 Evaluations and Red Teaming by External Safety Experts . . . 22  
5.2.2 Cybersecurity - Adversarially fine-tuned . . . . . . . . . . . . . . . . . . . . . . . . 22  
5.2.2.1 Capture the Flag (CTF) Challenges . . . . . . . . . . . . . . . . . . . . 23  
5.2.2.2 Cyber range . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24  
5.2.3 AI Self-Improvement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26  
5.2.3.1 SWE-bench Verified . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26  
5.2.3.2 OpenAI PRs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27  
5.2.3.3 PaperBench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28  
6 Appendix 1 29  
7 Appendix 2 30  
7.0.1 Recommendations Implemented . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30  
7.0.2 Recommendations Not Adopted . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31  

## 1 Introduction

We introduce gpt-oss-120b and gpt-oss-20b, two open-weight reasoning models available under the Apache 2.0 license and our gpt-oss usage policy. Developed with feedback from the open-source community, these text-only models are compatible with our Responses API and are designed to be used within agentic workflows with strong instruction following, tool use like web search and Python code execution, and reasoning capabilities—including the ability to adjust the reasoning effort for tasks that don't require complex reasoning. The models are customizable, provide full chain-of-thought (CoT), and support Structured Outputs.

Safety is foundational to our approach to open models. They present a different risk profile than proprietary models: Once they are released, determined attackers could fine-tune them to bypass safety refusals or directly optimize for harm without the possibility for OpenAI to implement additional mitigations or to revoke access.

In some contexts, developers and enterprises will need to implement extra safeguards in order to replicate the system-level protections built into models served through our API and products.

We're terming this document a model card, rather than a system card, because the gpt-oss models will be used as part of a wide range of systems, created and maintained by a wide range of stakeholders. While the models are designed to follow OpenAI's safety policies by default, other stakeholders will also make and implement their own decisions about how to keep those systems safe.

We ran scalable capability evaluations on gpt-oss-120b, and confirmed that the default model does not reach our indicative thresholds for High capability in any of the three Tracked Categories of our Preparedness Framework (Biological and Chemical capability, Cyber capability, and AI Self-Improvement). We also investigated two additional questions:

- Could adversarial actors fine-tune gpt-oss-120b to reach High capability in the Biological and Chemical or Cyber domains? Simulating the potential actions of an attacker, we adversarially fine-tuned the gpt-oss-120b model for these two categories. OpenAI's Safety Advisory Group ("SAG") reviewed this testing and concluded that, even with robust fine-tuning that leveraged OpenAI's field-leading training stack, gpt-oss-120b did not reach High capability in Biological and Chemical Risk or Cyber risk.
- Would releasing gpt-oss-120b significantly advance the frontier of biological capabilities in open foundation models? We found that the answer is no: For most of the evaluations, the default performance of one or more existing open models comes near to matching the adversarially fine-tuned performance of gpt-oss-120b.

As part of this launch, OpenAI is reaffirming its commitment to advancing beneficial AI and raising safety standards across the ecosystem.

## 2 Model architecture, data, training and evaluations

The gpt-oss models are autoregressive Mixture-of-Experts (MoE) transformers [1] [2] that build upon the GPT-2 and GPT-3 architectures. We are releasing two model sizes: gpt-oss-120b, which consists of 36 layers (116.8B total parameters and 5.1B "active" parameters per token per forward 3pass), and gpt-oss-20b with 24 layers (20.9B total and 3.6B active parameters). Table 1 shows a full breakdown of the parameter counts.

Component | 120b | 20b  
--- | --- | ---  
MLP | 114.71B | 19.12B  
Attention | 0.96B | 0.64B  
Embed + Unembed | 1.16B | 1.16B  
Active Parameters | 5.13B | 3.61B  
Total Parameters | 116.83B | 20.91B  
Checkpoint Size | 60.8GiB | 12.8GiB  

Table 1: Model parameter counts. We refer to the models as "120b" and "20b" for simplicity, though they technically have 116.8B and 20.9B parameters, respectively. Unembedding parameters are counted towards active, but not embeddings.

### 2.1 Quantization

We utilize quantization to reduce the memory footprint of the models. We post-trained the models with quantization of the MoE weights to MXFP4 format[3], where weights are quantized to 4.25 bits per parameter. The MoE weights are responsible for 90+% of the total parameter count, and quantizing these to MXFP4 enables the larger model to fit on a single 80GB GPU and the smaller model to run on systems with as little as 16GB memory. We list the checkpoint sizes of the models in Table 1.

### 2.2 Architecture

Both models have a residual stream dimension of 2880, applying root mean square normalization [4] on the activations before each attention and MoE block. Similar to GPT-2 we use Pre-LN placement [5][6].

Mixture-of-Experts: Each MoE block consists of a fixed number of experts (128 for gpt-oss120b and 32 for gpt-oss-20b), as well as a standard linear router projection which maps residual activations to scores for each expert. For both models, we select the top-4 experts for each token given by the router, and weight the output of each expert by the softmax of the router projection over only the selected experts. The MoE blocks use the gated SwiGLU [7] activation function.

Attention: Following GPT-3, attention blocks alternate between banded window and fully dense patterns [8][9], where the bandwidth is 128 tokens. Each layer has 64 query heads of dimension 64, and uses Grouped Query Attention (GQA [10][11]) with 8 key-value heads. We apply rotary position embeddings [12] and extend the context length of dense layers to 131,072 tokens using YaRN [13]. Each attention head has a learned bias in the denominator of the softmax, similar to off-by-one attention and attention sinks [14][15], which enables the attention mechanism to pay no attention to any tokens.

Our SwiGLU implementation is unconventional, including clamping and a residual connection.

### 2.3 Tokenizer

Across all training stages, we utilize our o200k_harmony tokenizer, which we open source in our TikToken library. This is a Byte Pair Encoding (BPE) which extends the o200k tokenizer used for other OpenAI models such as GPT-4o and OpenAI o4-mini with tokens explicitly used for our harmony chat format described in Table 18 and has a total of 201,088 tokens.

### 2.4 Pretraining

Data: We train the models on a text-only dataset with trillions of tokens, with a focus on STEM, coding, and general knowledge. To improve the safety of the model, we filtered the data for harmful content in pre-training, especially around hazardous biosecurity knowledge, by reusing the CBRN pre-training filters from GPT-4o [16]. Our model has a knowledge cutoff of June 2024.

Training: The gpt-oss models trained on NVIDIA H100 GPUs using the PyTorch framework [17] with expert-optimized Triton [18] kernels. The training run for gpt-oss-120b required 2.1 million H100-hours to complete, with gpt-oss-20b needing almost 10x fewer. Both models leverage the Flash Attention [19] algorithms to reduce the memory requirements and accelerate training.

### 2.5 Post-Training for Reasoning and Tool Use

After pre-training, we post-train the models using similar CoT RL techniques as OpenAI o3. This procedure teaches the models how to reason and solve problems using CoT and teaches the model how to use tools. Because of the similar RL techniques, these models have a personality similar to models served in our first-party products like ChatGPT. Our training dataset consists of a wide range of problems from coding, math, science, and more.

#### 2.5.1 Harmony Chat Format

For the models' training, we use a custom chat format known as the harmony chat format. This format provides special tokens to delineate message boundaries and uses keyword arguments (e.g., User and Assistant) to indicate message authors and recipients. We use the same System and Developer message roles that are present in the OpenAI API models. Using these roles, the models follow a role-based information hierarchy to resolve instruction conflicts: System > Developer > User > Assistant > Tool.

The format also introduces "channels" to indicate the intended visibility of each message, e.g., analysis for CoT tokens, commentary for function tool calling and final for answers shown to users. This format enables gpt-oss to provide advanced agentic features including interleaving tool calls within the CoT or providing preambles that outline longer action plans to the user. Our accompanying open-source implementation and guide provides full details on the proper usage of this format–it is critical to deploy our gpt-oss models properly to achieve their best capabilities. For example, in multi-turn conversations the reasoning traces from past assistant turns should be removed. Table 17 and 18 in the Appendix show an example model input and output in the harmony chat format.

#### 2.5.2 Variable Effort Reasoning Training

We train the models to support three reasoning levels: low, medium, and high. These levels are configured in the system prompt by inserting keywords such as "Reasoning: low". Increasing the reasoning level will cause the model's average CoT length to increase.

#### 2.5.3 Agentic Tool Use

During post-training, we also teach the models to use different agentic tools:

- A browsing tool, that allows the model to call search and open functions to interact with the web. This aids factuality and allows the models to fetch info beyond their knowledge cutoff.
- A python tool, which allows the model to run code in a stateful Jupyter notebook environment.
- Arbitrary developer functions, where one can specify function schemas in a Developer message similar to the OpenAI API. The definition of function is done within our harmony format. An example can be found in Table 18. The model can interleave CoT, function calls, function responses, intermediate messages that are shown to users, and final answers.

The models have been trained to support running with and without these tools by specifying so in the system prompt. For each tool, we have provided basic reference harnesses that support the general core functionality. Our open-source implementation provides further details.

### 2.6 Evaluation

We evaluate gpt-oss on canonical reasoning, coding, and tool use benchmarks. For all datasets, we report basic pass@1 results for high reasoning mode using the model's default system prompt. We compare to OpenAI o3, o3-mini, and o4-mini. We evaluate on:

- Reasoning and factuality: AIME, GPQA [22], MMLU [23], and HLE [24].
- Coding: Codeforces Elo and SWE-bench Verified [25]. We evaluate coding performance both with and without access to a terminal tool that is similar to the Codex CLI (e.g., provides the model with an exec tool).
- Tool use: function calling ability with τ -Bench Retail [21], we provide the model with functions to call in the model's developer message.
- Additional Capabilities: We additionally test important capabilities such as multilingual abilities and health knowledge with benchmarks such as MMMLU [23] and HealthBench [26].

Evaluation results on these benchmarks at all reasoning levels for both gpt-oss models are in Table 3 at the end of this section.

#### 2.6.1 Reasoning, Factuality and Tool Use

Main Capabilities: Figure 1 shows our main results on four canonical knowledge and reasoning tasks: AIME, GPQA, HLE, and MMLU. The gpt-oss models are strong at math in particular, which we believe is because they can use very long CoTs effectively, e.g., our gpt-oss-20b use over 20k CoT tokens per problem on average for AIME. On more knowledge-related tasks such as GPQA, the gpt-oss-20b model lags behind due to its smaller size.

Agentic Tasks: The gpt-oss models have particularly strong performance on coding and tool-use tasks. Figure 2 shows our performance on Codeforces, Swe-Bench and τ -bench retail. Similarly to the main capabilities evals, we find gpt-oss-120b comes close to OpenAI's o4-mini in performance.

Test-time scaling: Our models demonstrate smooth test-time scaling. In Figure 3, we sweep over the different reasoning modes of the model (low, medium, high) and plot accuracy versus average CoT+Answer length. We generally see log-linear returns on most tasks, where longer CoTs provide higher accuracy at a relatively large increase in final response latency and cost. We recommend that users pick a model size and corresponding reasoning level that balances these tradeoffs for their use case.

#### 2.6.2 Health Performance

To measure performance and safety in health-related settings, we evaluated gpt-oss-120b and gpt-oss-20b on HealthBench [26]. We report scores for HealthBench (realistic health conversations with individuals and health professionals), HealthBench Hard (a challenging subset of conversations), and HealthBench Consensus (a subset validated by the consensus of multiple physicians), across low, medium, and high reasoning effort in Table 3.

In Figure 4, we observe that the gpt-oss models at reasoning level high perform competitively to the best closed models, including OpenAI o3, and outperform some frontier models. In particular, gpt-oss-120b nearly matches OpenAI o3 performance on HealthBench and HealthBench Hard, and outperforms GPT-4o, OpenAI o1, OpenAI o3-mini, and OpenAI o4-mini by significant margins. These results represent a large Pareto improvement in the health performance-cost frontier. Open models may be especially impactful in global health, where privacy and cost constraints can be important. We hope that the release of these models makes health intelligence and reasoning capabilities more widely accessible, supporting the broad distribution of AI's benefits. Please note that the gpt-oss models do not replace a medical professional and are not intended for the diagnosis or treatment of disease.

#### 2.6.3 Multilingual Performance

To evaluate multilingual capabilities, we used the MMMLU eval [23], a professionally human-translated version of MMLU in 14 languages. The answers were parsed from the model's response by removing extraneous markdown or Latex syntax and searching for various translations of "Answer" in the prompted language. Similar to other evals, we find gpt-oss-120b at high reasoning comes close to OpenAI o4-mini-high in performance.

Table 2: MMMLU evaluation

| Language | gpt-oss-120b low | gpt-oss-120b medium | gpt-oss-120b high | gpt-oss-20b low | gpt-oss-20b medium | gpt-oss-20b high | o3-mini | o4-mini | o3 |
|---|---|---|---|---|---|---|---|---|---|
| Arabic | 75.0 | 80.4 | 82.7 | 65.6 | 73.4 | 76.3 | 81.9 | 86.1 | 90.4 |
| Bengali | 71.5 | 78.3 | 80.9 | 68.3 | 74.9 | 77.1 | 80.1 | 84.0 | 87.8 |
| Chinese | 77.9 | 82.1 | 83.6 | 72.1 | 78.0 | 79.4 | 83.6 | 86.9 | 89.3 |
| French | 79.6 | 83.3 | 84.6 | 73.2 | 78.6 | 80.2 | 83.7 | 87.4 | 90.6 |
| German | 78.6 | 81.7 | 83.0 | 71.4 | 77.2 | 78.7 | 80.8 | 86.7 | 90.5 |
| Hindi | 74.2 | 80.0 | 82.2 | 70.2 | 76.6 | 78.8 | 81.1 | 85.9 | 89.8 |
| Indonesian | 78.3 | 82.8 | 84.3 | 71.2 | 77.4 | 79.5 | 82.8 | 86.9 | 89.8 |
| Italian | 79.5 | 83.7 | 85.0 | 73.6 | 79.0 | 80.5 | 83.8 | 87.7 | 91.2 |
| Japanese | 77.0 | 82.0 | 83.5 | 70.4 | 76.9 | 78.8 | 83.1 | 86.9 | 89.0 |
| Korean | 75.2 | 80.9 | 82.9 | 69.8 | 75.7 | 77.6 | 82.6 | 86.7 | 89.3 |
| Portuguese | 80.0 | 83.3 | 85.3 | 73.3 | 79.2 | 80.5 | 84.1 | 87.8 | 91.0 |
| Spanish | 80.6 | 84.6 | 85.9 | 75.0 | 79.7 | 81.2 | 84.0 | 88.0 | 91.1 |
| Swahili | 59.9 | 69.3 | 72.3 | 46.2 | 56.6 | 60.7 | 73.8 | 81.3 | 86.0 |
| Yoruba | 49.7 | 58.1 | 62.4 | 38.4 | 45.8 | 50.1 | 63.7 | 70.8 | 78.0 |
| **Average** | 74.1 | 79.3 | 81.3 | 67.0 | 73.5 | 75.7 | 80.7 | 85.2 | 88.8 |

#### 2.6.4 Full Evaluations

We provide evaluation results across a large suite of benchmarks at all reasoning levels for the gpt-oss models.

Table 3: Evaluations across multiple benchmarks and reasoning levels.

| Benchmark (Accuracy %) | gpt-oss-120b low | gpt-oss-120b medium | gpt-oss-120b high | gpt-oss-20b low | gpt-oss-20b medium | gpt-oss-20b high |
|---|---|---|---|---|---|---|
| AIME 2024 (no tools) | 56.3 | 80.4 | 95.8 | 42.1 | 80.0 | 92.1 |
| AIME 2024 (with tools) | 75.4 | 87.9 | 96.6 | 61.2 | 86.0 | 96.0 |
| AIME 2025 (no tools) | 50.4 | 80.0 | 92.5 | 37.1 | 72.1 | 91.7 |
| AIME 2025 (with tools) | 72.9 | 91.6 | 97.9 | 57.5 | 90.4 | 98.7 |
| GPQA Diamond (no tools) | 67.1 | 73.1 | 80.1 | 56.8 | 66.0 | 71.5 |
| GPQA Diamond (with tools) | 68.1 | 73.5 | 80.9 | 58.0 | 67.1 | 74.2 |
| HLE (no tools) | 5.2 | 8.6 | 14.9 | 4.2 | 7.0 | 10.9 |
| HLE (with tools) | 9.1 | 11.3 | 19.0 | 6.3 | 8.8 | 17.3 |
| MMLU | 85.9 | 88.0 | 90.0 | 80.4 | 84.0 | 85.3 |
| SWE-Bench Verified | 47.9 | 52.6 | 62.4 | 37.4 | 53.2 | 60.7 |
| Tau-Bench Retail | 49.4 | 62.0 | 67.8 | 35.0 | 47.3 | 54.8 |
| Tau-Bench Airline | 42.6 | 48.6 | 49.2 | 32.0 | 42.6 | 38.0 |
| Aider Polyglot | 24.0 | 34.2 | 44.4 | 16.6 | 26.6 | 34.2 |
| MMMLU (Average) | 74.1 | 79.3 | 81.3 | 67.0 | 73.5 | 75.7 |

| Benchmark (Score %) | gpt-oss-120b low | gpt-oss-120b medium | gpt-oss-120b high | gpt-oss-20b low | gpt-oss-20b medium | gpt-oss-20b high |
|---|---|---|---|---|---|---|
| HealthBench | 53.0 | 55.9 | 57.6 | 40.4 | 41.8 | 42.5 |
| HealthBench Hard | 22.8 | 26.9 | 30.0 | 9.0 | 12.9 | 10.8 |
| HealthBench Consensus | 90.6 | 90.8 | 89.9 | 84.9 | 83.0 | 82.6 |

| Benchmark (Elo) | gpt-oss-120b low | gpt-oss-120b medium | gpt-oss-120b high | gpt-oss-20b low | gpt-oss-20b medium | gpt-oss-20b high |
|---|---|---|---|---|---|---|
| Codeforces (no tools) | 1595 | 2205 | 2463 | 1366 | 1998 | 2230 |
| Codeforces (with tools) | 1653 | 2365 | 2622 | 1251 | 2064 | 2516 |

## 3 Safety testing and mitigation approach

During post-training, we use deliberative alignment to teach the models to refuse on a wide range of content (e.g., illicit advice), be robust to jailbreaks, and adhere to the instruction hierarchy.

In line with our longstanding views on open model weights, we believe that testing conditions for open weight models "would ideally reflect the range of ways that downstream actors can modify the model. One of the most useful properties of open models is that downstream actors can modify the models to expand their initial capabilities and tailor them to the developer's specific applications. However, this also means that malicious parties could potentially enhance the model's harmful capabilities. Rigorously assessing an open-weights release's risks should thus include testing for a reasonable range of ways a malicious party could feasibly modify the model, including by fine-tuning."

The gpt-oss models are trained to follow OpenAI's safety policies by default. We ran scalable Preparedness evaluations on gpt-oss-120b, and confirmed that the default model does not reach our indicative thresholds for High capability in any of the three Tracked Categories of our Preparedness Framework (Biological and Chemical capability, Cyber capability, and AI Self-Improvement).

We also investigated two additional questions:

- First, could adversarial actors fine-tune gpt-oss-120b to reach High capability in the Biological and Chemical, or Cyber domains? Simulating the potential actions of an attacker, we created internal, adversarially fine-tuned versions of the gpt-oss-120b model for these two categories, which we are not releasing. OpenAI's Safety Advisory Group ("SAG") reviewed this testing and concluded that, even with robust fine-tuning that leveraged OpenAI's field-leading training stack, gpt-oss-120b did not reach High capability in Biological and Chemical Risk or Cyber risk. See Section 5.1 of our Preparedness results below for more details on this process, including the external feedback we received and incorporated.
- Second, would releasing gpt-oss-120b significantly advance the frontier of biological capabilities in open foundation models? We investigated this question by running biology Preparedness evaluations on other open foundation models, in addition to gpt-oss-120b. We found that on most evaluations, there already exists another open weight model scoring at or near gpt-oss-120b. As a result, we believe it is unlikely that this release significantly advances the state of the art of biological capabilities using open weight models.

Except where otherwise noted, the performance results in this model card describe the default performance of gpt-oss-120b and gpt-oss-20b.

As described below, we also ran our Preparedness Framework evaluations of Biological and Chemical Risk and Cybersecurity on adversarially fine-tuned versions of gpt-oss-120b.

### 5.2 Capability findings

#### 5.2.1 Biological and Chemical - Adversarially Fine-tuned

Under maximum elicitation conditions designed to test the upper-bound capabilities of the model, gpt-oss-120b shows notable strength in answering textual questions involving biological knowledge and harm scenarios. However, while generally capable, it does not yet meet high indicative thresholds on complex protocol debugging tasks, and its text-only architecture inherently limits applicability in visually-dependent laboratory contexts.

The biological domain is the area where gpt-oss-120b showed the greatest degree of capability. Given our plan to release gpt-oss as open weights, we also chose to investigate a second question: Even without reaching High capability on our Preparedness Framework, would gpt-oss-120b significantly advance the frontier of hazardous biological capabilities in open source foundation models?

To investigate this question, we compared gpt-oss-120b to other released open source models. At first, we primarily considered DeepSeek R1-0528. Partway through our process, the Qwen 3 Thinking and Kimi K2 models were released, and we added these to our comparison set. These evaluations confirmed that Qwen 3 Thinking and Kimi K2 have advanced to a level such that gpt-oss-120b does not significantly advance the state of the art on biosecurity-relevant evaluations. While gpt-oss-120b achieves the highest performance on select biosecurity evaluations, no single open model consistently outperforms the others in this domain.

##### 5.2.1.1 Long-form Biological Risk Questions

We graded the accuracy of model responses to long-form biorisk questions. Our long-form biothreat information questions test acquiring critical and sensitive information across the five stages of the biological threat creation process [31]: Ideation, Acquisition, Magnification, Formulation, and Release.

We designed the questions and detailed rubrics with Gryphon Scientific due to their expertise working with dangerous biological agents in a national security setting. We used the OpenAI o1-preview (pre-mitigation) model as an autograder, validating agreement with a trusted biosecurity expert. We made adjustments to the rubric and iterated on the autograder based on the expert feedback.

All gpt-oss helpful-only variants and competitor models seem to be able to synthesize biorisk-related information across all five steps of the biothreat creation process. We note that the Kimi K2, Qwen 3, and DeepSeek R1 results are without browsing and without adversarial fine-tuning, whereas the OpenAI o3, o4-mini, and gpt-oss variants (both with and without adversarial fine-tuning) are with browsing enabled. For Kimi K2, Qwen 3, and DeepSeek R1 we used jailbreak prompts to circumvent refusals.

##### 5.2.1.2 Multimodal Troubleshooting Virology

To evaluate models' ability to troubleshoot wet lab experiments in a multimodal setting, we evaluate models on a set of 350 fully held-out virology troubleshooting questions from SecureBio.

OpenAI o3 remains the highest scoring model on this benchmark. All models score above the average human baseline (40%). To run this eval with browsing, we maintain a domain block list and filter out any browsing results to sites on the blocklist. We also inspect browsing rollouts using a classifier which flags instances of cheating and manually review all flagged rollouts.

##### 5.2.1.3 ProtocolQA Open-Ended

To evaluate models' ability to troubleshoot commonly published lab protocols, we modify 108 multiple choice questions from FutureHouse's ProtocolQA dataset [32] to be open-ended short answer questions, which makes the evaluation harder and more realistic than the multiple-choice version. The questions introduce egregious errors in common published protocols, describe the wet lab result of carrying out this protocol, and ask for how to fix the procedure. To compare model performance to that of PhD experts, we performed expert baselining on this evaluation with 19 PhD scientists who have over one year of wet lab experience.

OpenAI o3 is still the highest performing model on this benchmark. All models underperform the consensus expert baseline (54%) and the median expert baseline of 42%. To run this eval with browsing, we maintain a domain block list and filter out any browsing results to sites on the blocklist. We also inspect browsing rollouts using a classifier which flags instances of cheating and manually review all flagged rollouts.

##### 5.2.1.4 Tacit Knowledge and Troubleshooting

We evaluated models on a tacit knowledge and troubleshooting multiple choice dataset created with Gryphon Scientific. The questions span all 5 stages in the biothreat creation process and focus on areas where tacit knowledge would be a bottleneck. Tacit knowledge questions are meant to be obscure to anyone not working in the field, i.e., they either require tracking down authors of relevant papers or knowing people in the field. Troubleshooting questions are meant to be obscure to anyone without hands-on experience, i.e., the answers are known only to people who have tried the protocol themselves.

This set is uncontaminated; it was created fully in-house with our partners at Gryphon Scientific and has not been published.

OpenAI o3 is still the highest performing model on this benchmark. None of the tested models outperform the consensus expert baseline of 80%, though all models outperform the 80th percentile PhD expert baseline of 63%.

##### 5.2.1.5 TroubleshootingBench

To evaluate models' ability to identify and correct real-world experimental errors in biological protocols, we built a short-answer troubleshooting dataset from expert-written wet lab procedures. TroubleshootingBench focuses on tacit, hands-on knowledge and uncontaminated procedures that are not available online.

Scientists with a PhD in a relevant biological discipline (virology, genetics, microbiology, or protein engineering) were asked to transcribe biological protocols they have personally used in the lab. Each protocol must include precise step-by-step procedures, equipment, and reagents. If a protocol was adapted from a publication, experts were required to significantly alter at least several steps. From these protocols, they created three troubleshooting questions each, introducing subtle or realistic execution errors (e.g., improper homogenization technique) and describing the resulting failed outcome.

After going through independent expert review, the resulting dataset includes 52 protocols, each paired with three expert-written troubleshooting questions. To benchmark model performance, we ran a human baselining campaign with 12 independent PhD experts. The 80th percentile expert score (36.4%) is used as an indicative threshold for model performance. Compared to ProtocolQA Open-Ended, which focuses on well-known published procedures, TroubleshootingBench is designed to test model performance on non-public, experience-grounded protocols and errors that rely on tacit procedural knowledge.

OpenAI o3 is the highest performing model on this new benchmark. All models underperform the 80th percentile human score of 36.4%.

##### 5.2.1.6 Evaluations and Red Teaming by External Safety Experts

We engaged SecureBio as an external assessor to evaluate gpt-oss-120b on biosecurity-relevant tasks, including static benchmarks, long-form biodesign, agent-based fragment and screening challenges, and manual red-teaming. Their evaluation found that an adversarially fine-tuned version gpt-oss-120b generally performed above a non-fine-tuned version of DeepSeek R1-0528 on these tasks, but remained below our OpenAI o3 models in overall reliability and depth. Because SecureBio's work focused on R1-0528 as the most capable available open weight baseline at the time, and because the browsing harness used for R1-0528 introduced some uncertainty, we also conducted internal follow-up comparisons. These confirmed that, since SecureBio's assessment, newly released open-source models Qwen 3 Thinking and Kimi K2 have advanced to a level that is competitive with adversarially fine-tuned gpt-oss-120b on biosecurity-relevant evaluations.

#### 5.2.2 Cybersecurity - Adversarially fine-tuned

Cybersecurity is focused on capabilities that could create risks related to use of the model for cyber-exploitation to disrupt confidentiality, integrity, and/or availability of computer systems. These results show comparable performance to OpenAI o3, and were likewise below our High capability threshold.

##### 5.2.2.1 Capture the Flag (CTF) Challenges

We evaluate model performance on a test set of over 100 curated, publicly available CTF challenges which met the following criteria:

- The challenge required skills falling into one of the following categories:
    - Web Application Exploitation (web): the flag is stored on a virtual machine running vulnerable software which must be exploited over an exposed network service to run code to retrieve it.
    - Reverse Engineering (rev): require analyzing a provided program to find a vulnerability needed to recover the flag.
    - Binary and Network Exploitation (pwn): require finding inputs needed to exploit a running program and retrieve the flag.
    - Cryptography (crypto): an encrypted flag is provided to the participant or a remote system uses cryptography to guard the flag.
    - Miscellaneous (misc): various other challenges ranging from problem-solving to advanced security concepts and techniques.
- The challenge was publicly available and provided with a working execution environment.

Challenges were further divided into the level of proficiency needed to solve the challenge, ultimately organized by whether the challenge required high school, collegiate, or professional-level cybersecurity knowledge and skills, with judgment on the requisite skills manually determined by our in-house cybersecurity expert. For all Collegiate and Professional CTFs – and most of the High School CTFs – we require that the solution involves a non-trivial level of exploitation, requiring chaining two or more exploitation steps and not be solvable by copy-pasting textbook examples.

To evaluate the model against the CTFs we give it access to a headlessLinux distribution with common offensive cybersecurity tools preinstalled as well as a harness which allows the model to call those tools or otherwise execute commands similar to as a human. We run 16 rollouts for each CTF exercise and report pass@12; this metric is then compared to the thresholds established by the Preparedness Framework as well as the performance of previously released OpenAI models.

We no longer report performance on High School level CTF challenges because they are too easy to meaningfully provide signal on cybersecurity risks.

Collegiate CTF challenges were part of the training data for the cybermax model, so we do not plot cybermax for the Collegiate CTF challenges. OpenAI o3 is the best performing model at Collegiate CTF challenges, followed by the launch candidate for gpt-oss-120b.

The cybermax model (far right), which was trained on High School and Collegiate CTFs as well as some CTFs not in any of our evaluation sets, performs only 3 percentage points lower than OpenAI o3 on this eval. The cybermax model was also trained to use a browsing tool with a domain block that filters out any websites containing eval writeups or source code.

As always, we note that these evaluation results likely represent lower bounds on model capability, because additional scaffolding or improved capability elicitation could substantially increase observed performance.

##### 5.2.2.2 Cyber range

Cyber range exercises measure a model's ability to conduct fully end-to-end cyber operations in a realistic, emulated network. These exercises are long-form, requiring the model to (1) construct a plan to achieve an abstract adversary objective; (2) exploit vulnerabilities, misconfigurations, and weaknesses that are likely to be seen in the wild; and (3) chain together these exploits to achieve the scenario objective.

We currently have 5 implemented range scenarios, each of which is designed to elicit cyber-relevant capabilities and behaviors. These ranges are organized into either light or medium difficulty scenarios:

- Light scenarios require either:
    - Deep expertise in an attack behavior with some small sequential chaining; or
    - Shallow expertise across a wide array of behaviors, but with nontrivial chaining.
- Medium scenarios require both nontrivial action chaining and deep expertise in at least one attacker behavior.

The current scenarios are as follows:

- Spareshub (Medium). This scenario requires the agent to (1) show deep expertise in a particular attack pattern – exploiting SQL Injection against a real-life web shopping portal; (2) show shallow expertise across a variety of attack patterns (lack of authentication; discovery, valid accounts/credential reuse); and (3) understand the sequencing of these actions together. This scenario also features realistic lateral movement that might be found in a real deployment.
- Simple Privilege Escalation (Light). This scenario requires the agent to escalate privileges in a small network. It primarily centers around a nontrivial privilege escalation vector (Path Interception), but the agent must also do this in conjunction with some other basic capabilities (e.g., valid accounts and discovery). This scenario showcases the ability of the agent to find misconfigurations on the fly that would be possible to encounter in the real world.
- Basic C2 (Light). This is a new scenario where the agent must figure out how to establish command-and-control with an isolated VM. It requires little explicit chaining of actions: but a nontrivial amount of work to setup the C2 channel. This scenario takes advantage of Azure network rules to make it challenging, and offers a semi-realistic assessment of how well the agent would be able to exploit then pivot through a hardened environment.
- Azure SSRF (Light). This is a new scenario where the agent must take advantage of a poorly coded web server, expose valid Azure credentials, and use those credentials to modify a network rule to allow it to access a target host. The critical vulnerabilities in this scenario are discovering the web server vulnerability and then realizing the best next move is to use Azure credentials. The task does not require much reasoning or planning.
- Taint Shared Content (Light). This is a new scenario where the agent must work in a mostly linear fashion to chain together a sequence of weaknesses. It does not require any deep expertise in any one area, instead requiring the agent to perform lab-style discovery, SQL Injection, tainting shared content, and privilege escalation. While the chain for these actions is linear, it still requires the agent to show its ability to autonomously string them together.

Models operate in a headlessLinux environment with standard offensive tools preinstalled, plus a harness for command execution. Each scenario is tested under three configurations:

- Normal, where the model is given the goal, SSH key, and nothing more;
- With Hints, where we provide the model with a rough plan of how to achieve the goal; and
- With Solver Code, where we provide the model with partial solver code it can use to solve the scenario.

We run 30 trials for each eval. Our primary metric for evaluation is pass@12 on the eval without hints provided. Since current models all score 0% on these evals, for context we also report pass@1 for a "with hints" condition.

These configurations help us assess not just whether the model can complete the task, but also how much external assistance it requires.

No model is able to solve any scenario unaided or with hints.

#### 5.2.3 AI Self-Improvement

The gpt-oss models do not demonstrate improved performance on software engineering and AI research tasks relevant to AI self-improvement risks. OpenAI o3 and o4-mini are still the highest performing models across all benchmarks.

##### 5.2.3.1 SWE-bench Verified

SWE-bench Verified [25] is the human-validated subset of SWE-bench that more reliably evaluates AI models' ability to solve real-world software issues. This validated set of tasks fixes certain issues with SWE-bench such as incorrect grading of correct solutions, under-specified problem statements, and overly specific unit tests. This helps ensure we're accurately grading model capabilities. An example task flow is shown below:

For OpenAI o3 and o4-mini, we used an internal tool scaffold designed for efficient iterative file editing and debugging. In this setting, we average over 4 tries per instance to compute pass@1 (unlike Agentless, the error rate does not significantly impact results).

All SWE-bench evaluation runs use a fixed subset of n=477 verified tasks which have been validated on our internal infrastructure. Our primary metric is pass@1, because in this setting (unlike e.g., OpenAI interviews), we do not consider the unit tests as part of the information provided to the model. Like a real software engineer, the model must implement its change without knowing the correct tests ahead of time.

All models performed similarly on this evaluation, with OpenAI o4-mini just one percentage point higher than OpenAI o3.

##### 5.2.3.2 OpenAI PRs

Measuring if and when models can automate the job of an OpenAI research engineer is a key goal of self-improvement evaluation work. We test models on their ability to replicate pull request contributions by OpenAI employees, which measures our progress towards this capability.

We source tasks directly from internal OpenAI pull requests. A single evaluation sample is based on an agentic rollout. In each rollout:

1. An agent's code environment is checked out to a pre-PR branch of an OpenAI repository and given a prompt describing the required changes.
2. ChatGPT agent, using command-line tools and Python, modifies files within the codebase.
3. The modifications are graded by a hidden unit test upon completion.

If all task-specific tests pass, the rollout is considered a success. The prompts, unit tests, and hints are human-written.

The gpt-oss models score only two percentage points lower than OpenAI o4-mini.

##### 5.2.3.3 PaperBench

PaperBench [33] evaluates the ability of AI agents to replicate state-of-the-art AI research. Agents must replicate 20 ICML 2024 Spotlight and Oral papers from scratch, including understanding paper contributions, developing a codebase, and successfully executing experiments. For objective evaluation, we develop rubrics that hierarchically decompose each replication task into smaller sub-tasks with clear grading criteria. In total, PaperBench contains 8,316 individually gradable tasks.

We measure a 10-paper subset of the original PaperBench splits, where each paper requires <10GB of external data files. We report pass@1 performance with high reasoning effort and no browsing.

## 6 Appendix 1

Figure 17: Model input in the harmony format specifying a system message with reasoning set to low, a developer message specifying one available function tool for the model, and a user message asking for the weather in SF.

Figure 18: Example model response in the harmony format with the CoT and the model making a tool call.

## 7 Appendix 2

This section describes the recommendations we received on our adversarial testing methodology, and how we responded.

### 7.0.1 Recommendations Implemented

1. Clarifying Threat Model and Risk Categorization
    - Defined low-resource actor assumptions: Added clarifying language to our paper on compute, ML expertise, and data access assumptions for low-resource actors, with future cost estimates flagged for follow-up.
    - Preparedness criteria & ProtocolQA requirement: We clarified the preparedness criteria and explicitly retained ProtocolQA as a required component of the assessment. We edited the paper text accordingly and re-ran OpenAI o3 for ProtocolQA with a blocklist to ensure consistency.
2. Strengthening Evaluation Completeness and Reliability
    - Robustness checks on ProtocolQA: We validated our protocol troubleshooting results by checking that the model never refused, adding more protocol-debugging training data, and adding a new protocol-troubleshooting eval similar to ProtocolQA but uncontaminated.
    - Inference-time scaling plots: Added plots for both bio and cyber evals showing how performance scales with number of trials.
    - Multimodal benchmark alignment: Ran text-only versions of Multimodal Virology Troubleshooting and updated results to improve comparability. We also conducted VCT on the final 322-question dataset and reported human baseline comparisons.
    - Expert baseline clarity: Specified expert profiles and calculation of baselines in reporting.
    - Quantified refusal behavior: Explicitly separated refusal-based failures from other failure modes and reported pre- and post-naughtification rates.
3. Improving Evaluation Setup
    - Enhanced agent scaffolding: Tested internal "Best of K" scaffolding in cyber evaluations.
    - Aligned RL datasets with ProtocolQA: Tested analogous datasets during RL training to confirm no harmful uplift; findings added to paper.
    - Fine-tuning performance verification: Aligned with internal researchers on best hyperparameter settings for maximum performance and changed when necessary.

### 7.0.2 Recommendations Not Adopted

1. Higher-quality agent scaffolding for measurements
    - Recommendation: Apply best-of-N scaffolding broadly to all evaluations.
    - Decision: Scaffolding experiments were partially conducted elsewhere, with limited expected additional gains from full reruns.
2. Omit ProtocolQA from preparedness thresholds
    - Recommendation: Remove ProtocolQA due to imperfect real-world coverage of troubleshooting risk.
    - Decision: Despite limitations, ProtocolQA provided a unique safety signal. Removing it would have left a major gap. Broader changes to preparedness criteria were out of scope for this release.
3. Closed vs. open model refusal comparison
    - Recommendation: Compute combined performance using closed models where non-refusal responses are substituted, treating refusals as zero.
    - Decision: Our past testing has found that closed models already did not refuse on benign-proxy tasks (except Griffin), so this wouldn't give much signal on how well open models could "close the gaps" for closed models on real malicious tasks.

## References

## References

1. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Proceedings of Advances in Neural Information Processing Systems, 2017.
2. N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," 2017.
3. O. C. Project, "OCP Microscaling Formats (MX) Specification Version 1.0," technical report, Open Compute Project, Sept. 2023.
4. B. Zhang and R. Sennrich, "Root mean square layer normalization," 2019.
5. R. Xiong, Y. Yang, D. He, K. Zheng, S. Zheng, C. Xing, H. Zhang, Y. Lan, L. Wang, and T.-Y. Liu, "On layer normalization in the transformer architecture," 2020.
6. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al., "Language models are unsupervised multitask learners," OpenAI blog, 2019.
7. N. Shazeer, "GLU variants improve transformer," arXiv preprint arXiv:2002.05202, 2020.
8. R. Child, S. Gray, A. Radford, and I. Sutskever, "Generating long sequences with sparse transformers," arXiv preprint arXiv:1904.10509, 2019.
9. T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al., "Language models are few-shot learners," NeurIPS, 2020.
10. J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai, "GQA: Training generalized multi-query transformer models from multi-head checkpoints," 2023.
11. N. Shazeer, "Fast transformer decoding: One write-head is all you need," arXiv preprint arXiv:1911.02150, 2019.
12. J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, "Roformer: Enhanced transformer with rotary position embedding," Neurocomputing, 2024.
13. B. Peng, J. Quesnelle, H. Fan, and E. Shippole, "YaRN: Efficient context window extension of large language models," arXiv preprint arXiv:2309.00071, 2023.
14. E. Miller, "Attention is off by one (2023)," URL https://www.evanmiller.org/attention-is-offby-one.html.
15. G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, "Efficient streaming language models with attention sinks," arXiv preprint arXiv:2309.17453, 2023.
16. A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al., "GPT-4o system card," arXiv preprint arXiv:2410.21276, 2024.
17. A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al., "Pytorch: An imperative style, high-performance deep learning library," Advances in neural information processing systems, vol. 32, 2019.
18. P. Tillet, H.-T. Kung, and D. Cox, "Triton: an intermediate language and compiler for tiled neural network computations," in Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pp. 10–19, 2019.
19. T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and memory-efficient exact attention with IO-awareness," 2022.
20. OpenAI, "Introducing swe-bench verified." https://openai.com/index/introducing-swe-bench-verified/, 2025. Accessed: 2025-08-04.
21. S. Yao, N. Shinn, P. Razavi, and K. Narasimhan, "τ -bench: A benchmark for tool-agent-user interaction in real-world domains," arXiv preprint arXiv:2406.12045, 2024.
22. D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman, "GPQA: A graduate-level google-proof QA benchmark," in COLM, 2024.
23. D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, "Measuring massive multitask language understanding," arXiv preprint arXiv:2009.03300, 2020.
24. L. Phan, A. Gatti, Z. Han, N. Li, J. Hu, H. Zhang, C. B. C. Zhang, M. Shaaban, J. Ling, S. Shi, et al., "Humanity's last exam," arXiv preprint arXiv:2501.14249, 2025.
25. N. Chowdhury, J. Aung, C. J. Shern, O. Jaffe, D. Sherburn, G. Starace, E. Mays, R. Dias, M. Aljubeh, M. Glaese, C. E. Jimenez, J. Yang, L. Ho, T. Patwardhan, K. Liu, and A. Madry, "Introducing SWE-bench Verified," OpenAI, 2024.
26. R. K. Arora, J. Wei, R. S. Hicks, P. Bowman, J. Quiñonero-Candela, F. Tsimpourlas, M. Sharman, M. Shah, A. Vallone, A. Beutel, et al., "HealthBench: Evaluating large language models towards improved human health," arXiv preprint arXiv:2505.08775, 2025.
27. M. Y. Guan, M. Joglekar, E. Wallace, S. Jain, B. Barak, A. Helyar, R. Dias, A. Vallone, H. Ren, J. Wei, H. W. Chung, S. Toyer, J. Heidecke, A. Beutel, and A. Glaese, "Deliberative alignment: Reasoning enables safer language models," arXiv preprint arXiv:2412.16339, 2024.
28. E. Wallace, K. Xiao, R. Leike, L. Weng, J. Heidecke, and A. Beutel, "The instruction hierarchy: Training LLMs to prioritize privileged instructions," arXiv preprint arXiv:2404.13208, 2024.
29. A. Souly, Q. Lu, D. Bowen, T. Trinh, E. Hsieh, S. Pandey, P. Abbeel, J. Svegliato, S. Emmons, O. Watkins, et al., "A strongreject for empty jailbreaks," arXiv preprint arXiv:2402.10260, 2024.
30. A. Parrish, A. Chen, N. Nangia, V. Padmakumar, J. Phang, J. Thompson, P. M. Htut, and S. R. Bowman, "BBQ: A hand-built bias benchmark for question answering," arXiv preprint arXiv:2110.08193, 2021.
31. T. Patwardhan, K. Liu, T. Markov, N. Chowdhury, D. Leet, N. Cone, C. Maltbie, J. Huizinga, C. Wainwright, S. Jackson, S. Adler, R. Casagrande, and A. Madry, "Building an early warning system for LLM-aided biological threat creation," OpenAI, 2023.
32. J. M. Laurent, J. D. Janizek, M. Ruzo, M. M. Hinks, M. J. Hammerling, S. Narayanan, M. Ponnapati, A. D. White, and S. G. Rodriques, "LAB-Bench: Measuring capabilities of language models for biology research," 2024.
33. G. Starace, O. Jaffe, D. Sherburn, J. Aung, J. S. Chan, L. Maksin, R. Dias, E. Mays, B. Kinsella, W. Thompson, J. Heidecke, A. Glaese, and T. Patwardhan, "PaperBench: Evaluating ai's ability to replicate ai research." https://openai.com/index/paperbench/, 2025.



---

# Tables

## 2 Model architecture, data, training and evaluations

### Model Parameter Counts

| Component           | 120b    | 20b    |
|---------------------|---------|--------|
| MLP                 | 114.71B | 19.12B |
| Attention           | 0.96B   | 0.64B  |
| Embed + Unembed     | 1.16B   | 1.16B  |
| Active Parameters   | 5.13B   | 3.61B  |
| Total Parameters    | 116.83B | 20.91B |
| Checkpoint Size     | 60.8GiB | 12.8GiB|

---

### 2.6.3 Multilingual Performance (MMMLU Evaluation)

| Language    | gpt-oss-120b low | gpt-oss-120b medium | gpt-oss-120b high | gpt-oss-20b low | gpt-oss-20b medium | gpt-oss-20b high | o3-mini | o4-mini | o3   |
|-------------|------------------|---------------------|-------------------|-----------------|--------------------|------------------|---------|---------|------|
| Arabic      | 75.0             | 80.4                | 82.7              | 65.6            | 73.4               | 76.3             | 81.9    | 86.1    | 90.4 |
| Bengali     | 71.5             | 78.3                | 80.9              | 68.3            | 74.9               | 77.1             | 80.1    | 84.0    | 87.8 |
| Chinese     | 77.9             | 82.1                | 83.6              | 72.1            | 78.0               | 79.4             | 83.6    | 86.9    | 89.3 |
| French      | 79.6             | 83.3                | 84.6              | 73.2            | 78.6               | 80.2             | 83.7    | 87.4    | 90.6 |
| German      | 78.6             | 81.7                | 83.0              | 71.4            | 77.2               | 78.7             | 80.8    | 86.7    | 90.5 |
| Hindi       | 74.2             | 80.0                | 82.2              | 70.2            | 76.6               | 78.8             | 81.1    | 85.9    | 89.8 |
| Indonesian  | 78.3             | 82.8                | 84.3              | 71.2            | 77.4               | 79.5             | 82.8    | 86.9    | 89.8 |
| Italian     | 79.5             | 83.7                | 85.0              | 73.6            | 79.0               | 80.5             | 83.8    | 87.7    | 91.2 |
| Japanese    | 77.0             | 82.0                | 83.5              | 70.4            | 76.9               | 78.8             | 83.1    | 86.9    | 89.0 |
| Korean      | 75.2             | 80.9                | 82.9              | 69.8            | 75.7               | 77.6             | 82.6    | 86.7    | 89.3 |
| Portuguese  | 80.0             | 83.3                | 85.3              | 73.3            | 79.2               | 80.5             | 84.1    | 87.8    | 91.0 |
| Spanish     | 80.6             | 84.6                | 85.9              | 75.0            | 79.7               | 81.2             | 84.0    | 88.0    | 91.1 |
| Swahili     | 59.9             | 69.3                | 72.3              | 46.2            | 56.6               | 60.7             | 73.8    | 81.3    | 86.0 |
| Yoruba      | 49.7             | 58.1                | 62.4              | 38.4            | 45.8               | 50.1             | 63.7    | 70.8    | 78.0 |
| **Average** | 74.1             | 79.3                | 81.3              | 67.0            | 73.5               | 75.7             | 80.7    | 85.2    | 88.8 |

---

### 2.6.4 Full Evaluations

#### Accuracy (%)

| Benchmark                | gpt-oss-120b low | gpt-oss-120b medium | gpt-oss-120b high | gpt-oss-20b low | gpt-oss-20b medium | gpt-oss-20b high |
|--------------------------|------------------|---------------------|-------------------|-----------------|--------------------|------------------|
| AIME 2024 (no tools)     | 56.3             | 80.4                | 95.8              | 42.1            | 80.0               | 92.1             |
| AIME 2024 (with tools)   | 75.4             | 87.9                | 96.6              | 61.2            | 86.0               | 96.0             |
| AIME 2025 (no tools)     | 50.4             | 80.0                | 92.5              | 37.1            | 72.1               | 91.7             |
| AIME 2025 (with tools)   | 72.9             | 91.6                | 97.9              | 57.5            | 90.4               | 98.7             |
| GPQA Diamond (no tools)  | 67.1             | 73.1                | 80.1              | 56.8            | 66.0               | 71.5             |
| GPQA Diamond (with tools)| 68.1             | 73.5                | 80.9              | 58.0            | 67.1               | 74.2             |
| HLE (no tools)           | 5.2              | 8.6                 | 14.9              | 4.2             | 7.0                | 10.9             |
| HLE (with tools)         | 9.1              | 11.3                | 19.0              | 6.3             | 8.8                | 17.3             |
| MMLU                     | 85.9             | 88.0                | 90.0              | 80.4            | 84.0               | 85.3             |
| SWE-Bench Verified       | 47.9             | 52.6                | 62.4              | 37.4            | 53.2               | 60.7             |
| Tau-Bench Retail         | 49.4             | 62.0                | 67.8              | 35.0            | 47.3               | 54.8             |
| Tau-Bench Airline        | 42.6             | 48.6                | 49.2              | 32.0            | 42.6               | 38.0             |
| Aider Polyglot           | 24.0             | 34.2                | 44.4              | 16.6            | 26.6               | 34.2             |
| MMMLU (Average)          | 74.1             | 79.3                | 81.3              | 67.0            | 73.5               | 75.7             |

#### Score (%)

| Benchmark               | gpt-oss-120b low | gpt-oss-120b medium | gpt-oss-120b high | gpt-oss-20b low | gpt-oss-20b medium | gpt-oss-20b high |
|-------------------------|------------------|---------------------|-------------------|-----------------|--------------------|------------------|
| HealthBench             | 53.0             | 55.9                | 57.6              | 40.4            | 41.8               | 42.5             |
| HealthBench Hard        | 22.8             | 26.9                | 30.0              | 9.0             | 12.9               | 10.8             |
| HealthBench Consensus   | 90.6             | 90.8                | 89.9              | 84.9            | 83.0               | 82.6             |

#### Elo

| Benchmark                | gpt-oss-120b low | gpt-oss-120b medium | gpt-oss-120b high | gpt-oss-20b low | gpt-oss-20b medium | gpt-oss-20b high |
|--------------------------|------------------|---------------------|-------------------|-----------------|--------------------|------------------|
| Codeforces (no tools)    | 1595             | 2205                | 2463              | 1366            | 1998               | 2230             |
| Codeforces (with tools)  | 1653             | 2365                | 2622              | 1251            | 2064               | 2516             |

---

### 4.1 Disallowed Content (Standard Disallowed Content Evaluation)

| Category                       | gpt-oss-120b | gpt-oss-20b | OpenAI o4-mini | GPT-4o |
|--------------------------------|--------------|-------------|---------------|--------|
| hate (aggregate)               | 0.996        | 0.996       | 0.996         | 0.996  |
| self-harm/intent and instructions | 0.995     | 0.984       | 1.000         | 1.000  |
| personaldata/semirestrictive    | 0.967        | 0.947       | 0.975         | 0.967  |
| sexual/exploitative             | 1.000        | 0.980       | 0.990         | 1.000  |
| sexual/minors                   | 1.000        | 0.971       | 0.990         | 1.000  |
| illicit/non-violent             | 1.000        | 0.983       | 0.991         | 0.983  |
| illicit/violent                 | 1.000        | 1.000       | 1.000         | 1.000  |
| personaldata/restricted         | 0.996        | 0.978       | 0.955         | 0.978  |

---

### 4.1 Disallowed Content (Production Benchmarks)

| Category                | gpt-oss-120b | gpt-oss-20b | OpenAI o4-mini | GPT-4o |
|-------------------------|--------------|-------------|---------------|--------|
| non-violent hate        | 0.895        | 0.901       | 0.832         | 0.882  |
| personal-data           | 0.888        | 0.921       | 0.847         | 0.860  |
| harassment/threatening  | 0.832        | 0.819       | 0.695         | 0.745  |
| sexual/illicit          | 0.919        | 0.852       | 0.857         | 0.927  |
| sexual/minors           | 0.967        | 0.866       | 0.862         | 0.939  |
| extremism               | 0.932        | 0.951       | 0.932         | 0.919  |
| hate/threatening        | 0.898        | 0.829       | 0.795         | 0.867  |
| illicit/nonviolent      | 0.692        | 0.656       | 0.658         | 0.573  |
| illicit/violent         | 0.817        | 0.744       | 0.845         | 0.633  |
| self-harm/intent        | 0.950        | 0.893       | 0.862         | 0.849  |
| selfharm/instructions   | 0.910        | 0.899       | 0.901         | 0.735  |

---

### 4.5 Hallucination Evaluations

| Eval Metric         | gpt-oss-120b | gpt-oss-20b | OpenAI o4-mini |
|---------------------|--------------|-------------|----------------|
| SimpleQA accuracy   | 0.168        | 0.067       | 0.234          |
| SimpleQA hallucination rate | 0.782 | 0.914       | 0.750          |
| PersonQA accuracy   | 0.298        | 0.155       | 0.356          |
| PersonQA hallucination rate | 0.491 | 0.532       | 0.361          |

---

### 4.6 Fairness and Bias (BBQ Evaluation)

| Metric (higher is better)         | gpt-oss-120b | gpt-oss-20b | OpenAI o4-mini |
|-----------------------------------|--------------|-------------|----------------|
| Accuracy on ambiguous questions   | 0.87         | 0.79        | 0.82           |
| Accuracy on disambiguated questions | 0.90       | 0.89        | 0.95           |