# thermoPredict

Composition-based prediction of standard thermodynamic properties (V°, ΔfG°, ΔfH°, S°, Cp°)
of cement hydrates at 25 °C, with a provenance-tracked 222-phase database, per-subgroup
empirical uncertainties, and direct export to GEMS/ThermoFun/Reaktoro.

- Database build: `py scripts/build_database.py` (sources in `data/sources/`, citations in `data/REFERENCES.md`)
- Train & predict: see Korean usage guide below

---

# Python Thermodynamic Property Predictor

시멘트 수화물 조성을 입력하면 `V0d`, `G0d`, `H0d`, `S0d`, `Cp0d`를 Python/scikit-learn 모델로 예측하는 MVP입니다.

## Setup

```powershell
py -m pip install -r requirements.txt
```

## Train And Predict

UI로 실행:

```powershell
.\scripts\run_ui.ps1
```

브라우저에서 자동으로 Streamlit 화면이 열립니다. 기본 주소는 다음과 같습니다.

```text
http://localhost:8501
```

기본 예시 조성으로 실행:

```powershell
.\scripts\run_prediction.ps1
```

화학식 입력:

```powershell
.\scripts\run_prediction.ps1 -Formula "(CaO)1.5(SiO2)1(H2O)2.5"
```

원소 조성 직접 입력:

```powershell
.\scripts\run_prediction.ps1 -Composition "Ca=1.5,Si=1,O=6,H=5"
```

입력 파일에서 예측하고 결과 CSV 저장:

```powershell
.\scripts\run_prediction.ps1 -Validation kfold -InputFile "data\example_input_compositions.csv" -Output "prediction_results.csv"
```

입력 파일은 `Formula`, `Composition`, 또는 원소별 column을 사용할 수 있습니다. 예시는 [data/example_input_compositions.csv](data/example_input_compositions.csv)에 있습니다.

모델 후보군은 `fast`, `expanded`, `full` 중에서 고를 수 있습니다. 기본값은 `expanded`입니다.

```powershell
.\scripts\run_prediction.ps1 -Validation kfold -ModelSet expanded -TrainOnly
.\scripts\run_prediction.ps1 -Validation kfold -ModelSet full -TrainOnly
```

`full`은 Gaussian process 같은 느린 후보까지 포함합니다. leave-one-out과 함께 쓰면 시간이 오래 걸릴 수 있습니다.

모델 검증 결과만 보기:

```powershell
.\scripts\run_prediction.ps1 -TrainOnly
```

Python 모듈을 직접 실행해도 됩니다.

```powershell
py -m src.cli --formula "(CaO)1.5(SiO2)1(H2O)2.5"
```

## Database

- master DB(`data/selected_thermodynamic_properties.csv`, 196 rows)는
  `py scripts/build_database.py`로 재생성됩니다. 입력은
  `data/sources/cemdata18-thermofun.json`(공식 ThermoHub 배포본, 2026-07-12 취득)과
  `data/sources/legacy_master_2026-07-12.csv`(불변 스냅샷)입니다.
- 출처/인용 정리는 [data/REFERENCES.md](data/REFERENCES.md), 값 충돌 로그는
  `data/sources/value_conflicts.csv`를 보세요.
- 빌드 시 품질 규칙이 적용됩니다: legacy 10x 오타 보정, Water 행 표준값 대체,
  자리표시자 값(V0=0, Cp0=0, S0<=0)은 NaN 처리되어 학습에서 제외됩니다.

## Current Behavior

- `data/selected_thermodynamic_properties.csv`를 기본 학습 데이터로 사용합니다.
- valid row 전체를 학습 대상으로 사용합니다.
- feature는 `Ca, Si, Al, O, H, Na, Fe, Fe3, Fe2, S, S2, C, Cl, K, Mg, N` 순서를 사용합니다.
- 조성만으로 subgroup(C-S-H, C-A-S-H, CAH, AFt, AFm, Hydrogarnet, LDH, M-S-H, Zeolite 등)을
  규칙 기반으로 추론하고(`src/grouping.py`), one-hot feature로 추가합니다.
  `--no-group-features`로 끌 수 있습니다. 예측 결과에는 `inferred_subgroup`이 함께 출력됩니다.
- 학습 데이터의 `Subgroup` 컬럼이 있으면 그 값을 쓰고, 없으면 조성에서 추론합니다.
- 입력 파일 헤더의 `Fe3+`, `Fe³⁺`, `Fe(III)` 등은 자동으로 `Fe3`로 매핑됩니다(`Fe2`, `S2`도 동일).
- target은 `V0d`, `G0d`, `H0d`, `S0d`, `Cp0d` 5개입니다.
- leave-one-out 또는 5-fold validation으로 모델을 비교합니다.
- 후보 모델 세트:
  - `fast`: kNN, Ridge, quadratic Ridge, PLS, tree ensembles, boosting, SVR, kernel ridge,
    additivity 기반 모델(Additivity OLS, Additivity + group residual[/atom])
  - `expanded`: `fast` + ElasticNet, Lasso, Bayesian Ridge, ARD, Huber, Linear SVR, NuSVR, AdaBoost, RegressorChain, MLP, XGBoost
  - `full`: `expanded` + OMP, Passive Aggressive, SGD, Gaussian Process
- 기본 ranking metric은 target별 normalized RMSE 평균입니다.
- exact composition match가 있으면 모델 예측 대신 학습 데이터 값을 반환합니다.
- nearest training rows와 domain status를 함께 출력합니다.
- 예측값마다 경험적 불확실도(±)를 제공합니다: 선택된 모델의 cross-validation 잔차를
  subgroup별로 모아 |오차|의 중앙값과 95% 분위수를 계산합니다 (subgroup 표본이 5개
  미만이면 전체 잔차로 대체). batch 출력에는 `p95_*` 컬럼으로 추가됩니다.
- `--validation logo` (leave-one-subgroup-out)로 미지 계열에 대한 외삽 성능을 평가할 수 있습니다.

## Code Layout

- `src/data_loader.py`: CSV/Excel load and schema validation
- `src/features.py`: formula and direct composition parsing
- `src/models.py`: candidate scikit-learn models
- `src/evaluation.py`: cross validation and metrics
- `src/prediction.py`: final training, nearest rows, prediction
- `src/cli.py`: command-line interface
- `app.py`: Streamlit user interface
