
# VLMEvalKit 평가 실행 가이드

VLMEvalKit은 다양한 VLM(Vision-Language Model) 성능을 평가할 수 있는 통합 툴킷입니다. 이 문서는 로컬 또는 Hugging Face 기반 모델을 빠르게 테스트할 수 있도록 필수 절차만 간단히 정리한 가이드입니다. 보다 자세한 사항은 공식 GitHub를 참조하세요:

https://github.com/open-compass/VLMEvalKit

## 1. 설치 (Installation)

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

Gemma3 모델 관련 의존성 설치:

```bash
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```

# 설치 후 API 키 설정 (API Key Setup)

GPT-4V, Gemini-Pro-V 등의 API 모델로 추론하거나, LLM API를 정답 판별(Judge) 또는 선택 추출기로 사용할 경우, 먼저 API 키 설정이 필요합니다.

VLMEvalKit은 API 키가 설정되어 있을 경우 judge LLM을 통해 모델 출력에서 정답을 추출하며, 키가 없을 경우 정확 일치 모드(exact matching)를 사용합니다. 정확 일치 모드는 Yes/No 유형 및 객관식(Multi-choice) 문제에만 적용됩니다.

환경 변수로 직접 설정하거나 `$VLMEvalKit/.env` 파일을 생성하여 사용할 수 있습니다. `.env` 파일 예시는 다음과 같습니다:

```
# $VLMEvalKit/.env 파일 예시
# 상용 VLM API 키 설정

# QwenVL APIs
DASHSCOPE_API_KEY=

# Gemini (Google Cloud)
GOOGLE_API_KEY=

# OpenAI API
OPENAI_API_KEY=
OPENAI_API_BASE=

# StepAI API
STEPAI_API_KEY=

# REKA API
REKA_API_KEY=

# GLMV API
GLMV_API_KEY=

# CongRong API
CW_API_BASE=
CW_API_KEY=

# SenseNova API
SENSENOVA_API_KEY=

# Hunyuan-Vision API
HUNYUAN_SECRET_KEY=
HUNYUAN_SECRET_ID=

# LMDeploy API
LMDEPLOY_API_BASE=

# API 모델 호출 시 프록시 설정 (선택사항)
EVAL_PROXY=
```

빈 칸에 각자의 API 키를 입력하면 평가 실행 시 자동으로 불러와 사용됩니다.

## 2. 모델 설정 (Configuration)

`configs/models/config.py` 파일을 열어 사용할 모델을 설정합니다. 예를 들어 로컬에서 커스텀 학습한 Gemma3 모델을 사용하고자 할 경우 다음과 같이 수정합니다.

```python
gemma_series = {
    "paligemma-3b-mix-448": partial(
        PaliGemma, model_path="google/paligemma-3b-mix-448"
    ),
    "Gemma3-4B": partial(Gemma3, model_path="google/gemma-3-4b-it"),
    "Gemma3-12B": partial(Gemma3, model_path="google/gemma-3-12b-it"),
    "Gemma3-27B": partial(Gemma3, model_path="google/gemma-3-27b-it"),
    "Gemma3-Custom": partial(Gemma3, model_path="/path/to/your/custom_gemma3_model"),
}
```

※ 위의 `/path/to/your/custom_gemma3_model` 부분을 실제 로컬 모델 경로로 변경하십시오.

## 3. 평가 실행 (Evaluation)

기본 실행 명령어 예시입니다.

```bash
torchrun --nproc-per-node=2 run.py --data MME --model Gemma3-Custom --verbose
```

옵션 설명:
- `--data`: 평가할 벤치마크 이름
- `--model`: 설정한 모델 키 이름
- `--nproc-per-node`: 사용 GPU 수

## 4. 데이터셋 목록 (추천)

데이터셋은 `vlmeval/dataset` 경로에서 관리됩니다. 다음은 자주 사용하는 데이터셋 목록입니다.

### CharXiv
- CharXiv_descriptive_val
- CharXiv_reasoning_val

### image_vqa
- OCRVQA_TEST
- OCRVQA_TESTCORE
- TextVQA_VAL
- DocVQA_VAL
- InfoVQA_VAL
- ChartQA_TEST
- GQA_TestDev_Balanced

### image_caption
- COCO_VAL

### image_mcq
- RealWorldQA
- ScienceQA_VAL
- ScienceQA_TEST
- BLINK
- A-OKVQA
- AI2D_TEST
- AI2D_TEST_NO_MASK
- MMMU_DEV_VAL
- MMMU_TEST

## 5. 참고 사항

- 모델은 Hugging Face ID 또는 로컬 경로 사용 가능
- 평가 결과는 `outputs/` 폴더에 자동 저장
- 다중 GPU 환경에서는 `--nproc-per-node` 조정 필요

# Gemma3 Benchmark Results

https://huggingface.co/google/gemma-3-4b-it

## 1. Reasoning and Factuality

| Benchmark         | Metric     | Gemma 3 PT 1B | Gemma3-Custom | Gemma 3 PT 4B | Gemma 3 PT 12B | Gemma 3 PT 27B |
|------------------|------------|---------------|-----------|----------------|-----------------|------------------|
| HellaSwag        | 10-shot    | 62.3          |           | 77.2           | 84.2            | 85.6             |
| BoolQ            | 0-shot     | 63.2          |           | 72.3           | 78.8            | 82.4             |
| PIQA             | 0-shot     | 73.8          |           | 79.6           | 81.8            | 83.3             |
| SocialIQA        | 0-shot     | 48.9          |           | 51.9           | 53.4            | 54.9             |
| TriviaQA         | 5-shot     | 39.8          |           | 65.8           | 78.2            | 85.5             |
| Natural Questions| 5-shot     | 9.48          |           | 20.0           | 31.4            | 36.1             |
| ARC-c            | 25-shot    | 38.4          |           | 56.2           | 68.9            | 70.6             |
| ARC-e            | 0-shot     | 73.0          |           | 82.4           | 88.3            | 89.0             |
| WinoGrande       | 5-shot     | 58.2          |           | 64.7           | 74.3            | 78.8             |
| BIG-Bench Hard   | few-shot   | 28.4          |           | 50.9           | 72.6            | 77.7             |
| DROP             | 1-shot     | 42.4          |           | 60.1           | 72.2            | 77.2             |

## 2. STEM and Code

| Benchmark   | Metric   | Gemma3-Custom | Gemma 3 PT 4B | Gemma 3 PT 12B | Gemma 3 PT 27B |
|------------|----------|-----------|----------------|----------------|----------------|
| MMLU       | 5-shot   |           | 59.6           | 74.5           | 78.6           |
| MMLU (Pro COT) | 5-shot |       | 29.2           | 45.3           | 52.2           |
| AGIEval    | 3-5-shot |           | 42.1           | 57.4           | 66.2           |
| MATH       | 4-shot   |           | 24.2           | 43.3           | 50.0           |
| GSM8K      | 8-shot   |           | 38.4           | 71.0           | 82.6           |
| GPQA       | 5-shot   |           | 15.0           | 25.4           | 24.3           |
| MBPP       | 3-shot   |           | 46.0           | 60.4           | 65.6           |
| HumanEval  | 0-shot   |           | 36.0           | 45.7           | 48.8           |

## 3. Multilingual

| Benchmark         | Gemma 3 PT 1B | Gemma3-Custom | Gemma 3 PT 4B | Gemma 3 PT 12B | Gemma 3 PT 27B |
|------------------|----------------|-----------|----------------|-----------------|------------------|
| MGSM             | 2.04           |           | 34.7           | 64.3            | 74.3             |
| Global-MMLU-Lite | 24.9           |           | 57.0           | 69.4            | 75.7             |
| WMT24++ (ChrF)   | 36.7           |           | 48.4           | 53.9            | 55.7             |
| FloRes           | 29.5           |           | 39.2           | 46.0            | 48.8             |
| XQuAD (all)      | 43.9           |           | 68.0           | 74.5            | 76.8             |
| ECLeKTic         | 4.69           |           | 11.0           | 17.2            | 24.4             |
| IndicGenBench    | 41.4           |           | 57.2           | 61.7            | 63.4             |

## 4. Multimodal

| Benchmark        | Gemma3-Custom | Gemma 3 PT 4B | Gemma 3 PT 12B | Gemma 3 PT 27B |
|-----------------|-----------|----------------|----------------|----------------|
| COCOcap         |           | 102            | 111            | 116            |
| DocVQA (val)    |           | 72.8           | 82.3           | 85.6           |
| InfoVQA (val)   | 37.3          | 44.1           | 54.8           | 59.4           |
| MMMU (pt)       |           | 39.2           | 50.3           | 56.1           |
| TextVQA (val)   |           | 58.9           | 66.5           | 68.6           |
| RealWorldQA     |           | 45.5           | 52.2           | 53.9           |
| ReMI            |           | 27.3           | 38.5           | 44.8           |
| AI2D            |           | 63.2           | 75.2           | 79.0           |
| ChartQA         |           | 63.6           | 74.7           | 76.3           |
| VQAv2           |           | 63.9           | 71.2           | 72.9           |
| BLINK           |           | 38.0           | 35.9           | 39.6           |
| OKVQA           |           | 51.0           | 58.7           | 60.2           |
| TallyQA         |           | 42.5           | 51.8           | 54.3           |
| SpatialSense VQA|           | 50.9           | 60.0           | 59.4           |
| CountBenchQA    |           | 26.1           | 17.8           | 68.0           |

## 5. 추가 참고 사항

- 아래 멀티모달 벤치마크는 **VLMEvalKit에 기본 포함되어 있지 않으며**, 별도의 데이터 준비 및 커스텀 스크립트가 필요할 수 있습니다.
  - CountBenchQA
  - SpatialSense VQA
  - AREMI (ReMI)
  - TallyQA
