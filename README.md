
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

### CharXiv 시각 데이터셋
- CharXiv_descriptive_val
- CharXiv_reasoning_val

### image_vqa
- OCRVQA_TEST
- OCRVQA_TESTCORE
- TextVQA_VAL
- DocVQA_VAL
- DocVQA_TEST
- InfoVQA_VAL
- InfoVQA_TEST
- ChartQA_TEST
- GQA_TestDev_Balanced

### image_shortqa
- LiveMMBench_Infographic
- LiveMMBench_Perception
- LiveMMBench_Reasoning
- LiveMMBench_Reasoning_circular
- hle

### image_mcq
- RealWorldQA
- ScienceQA_VAL
- ScienceQA_TEST

## 5. 참고 사항

- 모델은 Hugging Face ID 또는 로컬 경로 사용 가능
- 평가 결과는 `outputs/` 폴더에 자동 저장
- 다중 GPU 환경에서는 `--nproc-per-node` 조정 필요
