
#  데이터셋 만들기 command
## 데이터셋 모으기
```
python experiments/robot/libero/run_libero_dataset.py \
--model_family openvla \
--pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
--task_suite_name libero_object \
--center_crop True \
--num_trials_per_task 5 \
--seed 42 \
--log_dataset True \
--dataset_dir ./rollouts_libero
```
## 여러 seed 돌리기##
```
#!/bin/bash
for seed in 42 43 44 45; do
    echo "Running with seed: $seed"
    python experiments/robot/libero/run_libero_dataset.py \
        --model_family openvla \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
        --task_suite_name libero_object \
        --center_crop True \
        --num_trials_per_task 40 \
        --seed $seed \
        --log_dataset True \
        --dataset_dir ./rollouts_libero \
        --run_id_note "seed_${seed}"
done
```

## 실패한 에피소드 찾기
```
~/choi_ws/openvla/rollouts/날짜
find . -type f -name "*success=False*" | sed -E 's/.*episode=([0-9]+).*/\1/'
```
## 데이터셋에 failure_tokens.npy 만들고 라벨링하기
```  
cd ~/choi_ws/openvla/rollouts_libeo  
python3 init_failure_tokens.py  
python3 set_failure_tokens.py  
python3 check_failure_token_npy.py  
```

# 학습 및 실행 command
## pretrain된 모델 처음 학습
```
cd ~/choi_ws/openvla/experiments/robot/failure_finetuning   
python train_failure_detection.py     --rollouts_dir ../../../rollouts_libero     --run_id LIBERO_object_dataset_selective     --batch_size 2     --learning_rate 1e-4     --num_epochs 10     --save_dir ./finetune_object_model

```

## best_model 체크포인트를 이어서 학습
```
cd ~/choi_ws/openvla/experiments/robot/failure_finetuning   
python train_failure_checkpoint.py \
    --checkpoint_path ./test_checkpoints/best_model \
    --rollouts_dir /home/choi/choi_ws/openvla/rollouts_libero \
    --run_id EVAL-libero_object-openvla-2025_08_22-14_51_38 \
    --learning_rate 5e-5 \
    --num_epochs 5
```

## eval
```
cd ~/choi_ws/openvla/experiments/robot/failure_finetuning   
python run_failure_eval.py \
    --model_checkpoint ./test_checkpoints/best_model \
    --task_suite_name libero_object \
    --center_crop True \
    --num_trials_per_task 5
```
# Goal
좋아, “전체 실패/성공을 예측(또는 인지)”하는 방향으로 내가 권하는 설계를 한꺼번에 정리해줄게. 핵심은 전역 실패 확률을 매 타임스텝마다 업데이트하게 해서, 네가 로그에서 본 “진동(앞뒤 왕복), 반복 집기, 진전 없음” 같은 패턴을 학습적으로 감지하도록 만드는 거야.

목표

매 스텝 t마다 **p_fail^global(t) = P(episode 실패 | 현재까지의 관측·행동 시퀀스)**를 예측.

이 값이 임계치 이상으로 일정 구간 유지되면 재지시/재계획/리트라이를 트리거.

포인트는 “그 스텝의 로컬 실패”가 아니라, 에피소드 전체가 실패로 갈 확률의 상승을 감지하는 것.

아키텍처(간단)

기존 VLA의 액션 토큰 출력 옆에 전역 실패 헤드(scalar or small token) 추가.

입력: 현재 시점까지의 언어·시각·행동 히스토리(autoregressive hidden state).

출력: p_fail^global(t) ∈ [0,1].

선택 옵션:

Time-to-failure 회귀 헤드도 함께 두기(남은 스텝 수 예측). p_fail보다 민감한 선행 신호가 됨.

“재계획 요청” 토큰을 별도로 두어, p_fail 상승 시 자체적으로 요청을 생성하게 하기(외부 루프를 단순화).

라벨 설계(학습 신호)

미래-결과 전파 라벨(가장 간단/강력)

에피소드가 실패로 종료되면, 모든 스텝 t에 대해 y_t = 1, 성공이면 y_t = 0.

이렇게 하면 모델이 “진동·반복·무진전” 패턴을 자동으로 실패 신호로 맵핑하도록 학습됨.

단, 너무 거칠 수 있으므로 아래 보조 라벨을 섞어 성능/안정화.

약지도(weak) 보조 라벨/피처로 “실패적 패턴” 강화

진동 점수: Δx,Δy,Δz,Δθ의 부호 전환 횟수/분산 증가.

무진전 지표: 목표까지의 거리가 일정 구간 동안 줄지 않음(visual pose/키포인트/scene graph 기반).

반복 집기/그리퍼 토글 빈도: open/close 패턴의 비정상적 반복.

행동-관측 불일치: 큰 액션에도 관측 변화가 미미.

충돌/슬립 감지: 접촉/힘 추정치 급등, 픽셀 유사도 급상승 후 원위치 등.

이 지표들을 self-supervised auxiliary loss로 사용하거나, y_t를 가중치 조정하는 데 활용.

손실 함수(안정화 포인트 포함)

기본: L = L_action + α·BCE(p_fail^global(t), y_t)

캘리브レー션: Brier loss를 소량 섞어 확률 해석력 개선.

시간적 매끈함: TV(총변동) 혹은 2차 차분 정규화로 p_fail의 잡음을 감소.

단조성(실패 에피소드 한정): 실패 episode에 대해 p_fail^global(t+1) ≥ p_fail^global(t)을 부드럽게 유도하는 페널티(soft hinge).
→ “점점 망해간다”는 신호를 학습적으로 받게 함.

선택: 하저드(hazard) 관점을 추가해 p_fail을 누적 생존확률과 일관되게 묶는 생존분석식 정규화.

인퍼런스 사용 규칙(루프 트리거)

히스테리시스로 튀는 값 억제:

p_fail ≥ τ_high가 K 스텝 연속이면 리트라이/재지시.

복귀는 p_fail ≤ τ_low M 스텝 연속일 때만.

행동 억제/수정: p_fail 급등 시 속도/가속 클리핑, 보수적 탐색 전환, 시야 재확보 동작 삽입 등.

데이터 구성(네 로그 기반)

기본 GT: 시뮬레이터(또는 로봇) 에피소드 최종 성공/실패 플래그.

전 스텝에 동일 라벨을 부여 → 모델이 “실패의 전조 패턴”을 통계적으로 학습.

보조 신호: 위 “진동·반복·무진전” 지표를 피처로 추출해 입력에 결합하거나, 보조 loss로 사용.

샘플링: 실패 에피소드의 초중후 구간을 골고루 oversample하여, 초기에 뜨는 전조 패턴도 인지하게 만듦.

검증: ROC-AUC, AUPRC 외에 리트라이 트리거 품질(precision@K, early-detection latency)을 별도로 측정.

장점 vs. 스텝-로컬 실패 토큰

너가 원한 대로, 이건 “전체 실패 인지”가 목적이라 에피소드의 거시적 패턴(진동, 반복 시도)을 더 잘 포착.

로컬 실패 판정과 달리, 지연된 실패(한참 돌다가 실패)도 일찍부터 위험도를 올릴 수 있음.

외부 모듈은 간단히 “p_fail 임계치 감시 + 재지시”만 하면 됨(검증기를 따로 붙일 필요가 크게 줄어듦).

확장 아이디어(필요 시)

듀얼 헤드: p_fail^global(t) + time_to_fail(t). 둘을 함께 쓰면 트리거 타이밍 최적화 용이.

대조학습: 정상 진행 구간 vs. 실패 전조 구간을 쌍으로 뽑아 시퀀스-레벨 대조 손실 추가.

언어 상태 반영: 재지시 이후 p_fail이 낮아지는 패턴까지 학습하도록, “프롬프트 리셋” 토큰을 입력 시퀀스에 명시.

원하는 바가 “에피소드 전역 실패를 타임스텝마다 갱신해서 빨리 인지하고, 임계치 넘으면 재시도”라면 위 구성이 가장 직관적이고 구현 난이도도 합리적이야.
다음 단계로, 네가 가진 LIBERO 로그에서 진동/무진전 지표를 자동 계산해 약지도 피처를 만들어 붙이는 파이프라인을 바로 설계해줄까, 아니면 우선 라벨만으로 학습하는 최소 버전부터 갈까?
