# Colab(T4) 측정 · 학교 RTX 4090 학습 안내

처음 하는 사람도 그대로 따라 할 수 있게 순서대로 적었습니다.

1. Windows PC에서 패치를 적용해 `perf/improve` 브랜치를 GitHub에 올립니다 (약 10분).
2. Google Colab에서 노트북을 열고 T4 GPU로 **모두 실행**합니다 (약 2시간).
3. `results.zip`을 팀에 보냅니다. 내 PC의 대시보드로 직접 볼 수도 있습니다.
4. (선택) 학교 Linux RTX 4090에서 학습 ablation을 돌립니다 (실험당 약 6~7시간).

무엇을 왜 재는지는 [성능 개선 계획](PERFORMANCE_PLAN.md)에 있습니다.

## 준비물

- 팀 저장소에 push할 수 있는 GitHub 계정, Colab·드라이브에 쓸 Google 계정
- Windows PC: [Git for Windows](https://git-scm.com/download/win)와 팀 저장소를 받아 둔 폴더 (아래 예시는 `C:\Users\82103\Desktop\Lite-R-CNN`)
- 전달받은 파일 두 개 (아래 예시는 `C:\Users\82103\Downloads`에 있다고 가정)
  - 패치 `lite-rcnn-perf-improve.patch`
  - 커밋 메시지 `COMMIT_MESSAGE_perf.txt`

## 1. Windows PC에서 브랜치 올리기

시작 메뉴에서 **명령 프롬프트**(cmd)를 열고, 아래 명령을 **한 줄씩** 붙여 넣고 Enter를 누릅니다
(한꺼번에 붙여 넣으면 앞 줄이 실패해도 다음 줄이 실행됩니다). 폴더나 파일 위치가 다르면 경로만 자기 것으로 바꾸십시오.

```bat
cd /d C:\Users\82103\Desktop\Lite-R-CNN
git status
git fetch origin
git switch -c perf/improve origin/main
git apply --index --check C:\Users\82103\Downloads\lite-rcnn-perf-improve.patch
git apply --index C:\Users\82103\Downloads\lite-rcnn-perf-improve.patch
git status
git commit -F C:\Users\82103\Downloads\COMMIT_MESSAGE_perf.txt
git push -u origin perf/improve
```

각 줄이 하는 일:

1. `cd /d …`: 저장소 폴더로 이동합니다 (`/d`는 명령 프롬프트가 다른 드라이브에서 열려도 이동하게 함).
2. `git status`: `modified:`로 표시된 파일이 없어야 합니다 (`Untracked files`는 괜찮음). 있으면 아래 표를 보세요.
3. `git fetch origin`: GitHub의 최신 `main`을 받아 옵니다.
4. `git switch -c perf/improve origin/main`: 최신 `main`에서 새 브랜치 `perf/improve`를 만들고 그 브랜치로 옮깁니다.
5. `git apply --index --check …`: 패치가 깨끗하게 들어가는지 확인만 합니다. **아무것도 출력되지 않으면 정상**입니다.
6. `git apply --index …`: 패치를 적용하고 바뀐 파일만 커밋 대상으로 올립니다 (`git add -A`를 쓰지 않으므로 내 컴퓨터의 다른 파일이 섞이지 않음).
7. `git status`: 바뀐 파일·새 파일이 `Changes to be committed`에 보이면 됩니다.
8. `git commit -F …`: 함께 받은 메시지 파일의 내용으로 커밋합니다.
9. `git push -u origin perf/improve`: 브랜치를 GitHub에 올립니다.

- 명령 뒤에 `# 설명`을 붙이지 마세요. 명령 프롬프트는 `#` 뒤의 글도 명령의 일부로 읽어서, 예를 들어 `git apply`가 `#`를 파일 이름으로 알고 실패합니다.
- 확인: GitHub 저장소 페이지의 브랜치 목록에 `perf/improve`가 있고, 그 안에 `notebooks/Lite_RCNN_T4_Colab.ipynb`가 보이면 성공입니다.

| 이런 메시지가 나오면 | 이렇게 하세요 |
|---|---|
| `'git'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는 배치 파일이 아닙니다.` | Git for Windows를 설치한 뒤 명령 프롬프트를 새로 엽니다 |
| `git status`에 `modified:` 파일이 보임 | 하던 작업을 먼저 커밋하거나, 새 폴더에 저장소를 다시 받아서 진행: `git clone https://github.com/AI-Engineering-Teamproject/Lite-R-CNN.git C:\Users\82103\Desktop\Lite-R-CNN-perf` → `cd /d C:\Users\82103\Desktop\Lite-R-CNN-perf` → `git fetch origin`부터 다시 |
| `git: 'switch' is not a git command` (옛 Git) | `git checkout -b perf/improve origin/main` |
| `a branch named 'perf/improve' already exists` | 다른 이름을 씁니다: `git switch -c perf/improve-2 origin/main`, 마지막 줄도 `git push -u origin perf/improve-2`. 노트북 설정 `BRANCH`도 같은 이름으로 |
| `git apply --check`가 `patch does not apply` | 그사이 `main`이 바뀐 것입니다. `git apply --index --3way C:\Users\82103\Downloads\lite-rcnn-perf-improve.patch`로 다시 시도하고, 충돌이 나면 멈추고 메시지를 보내 주세요 |
| `can't open patch` 또는 `could not read log file` | 파일 이름을 확인합니다: `dir C:\Users\82103\Downloads`. 브라우저가 `lite-rcnn-perf-improve (1).patch`처럼 이름을 바꿨다면 명령에 그 이름을 넣고, 빈칸이 있으니 큰따옴표로 감쌉니다: `git apply --index --check "C:\Users\82103\Downloads\lite-rcnn-perf-improve (1).patch"` |
| `Please tell me who you are` | 처음 한 번만 `git config --global user.name "이름"`과 `git config --global user.email "GitHub 메일"`을 실행한 뒤 `git commit` 줄부터 다시 |
| push할 때 로그인 창 | GitHub 계정으로 로그인합니다. `403`/권한 오류면 저장소 관리자에게 쓰기 권한을 요청하세요 |

패치 파일과 메시지 파일은 메모장 등으로 열어서 저장하지 마십시오. 줄바꿈이나 인코딩이 바뀌어 적용이 실패하거나 커밋 메시지가 깨질 수 있습니다.

## 2. Colab에서 측정하기

### 노트북 열기

1. <https://colab.research.google.com>에 접속해 Google 계정으로 로그인합니다.
2. 메뉴 **파일 → 노트북 열기**(File → Open notebook) → **GitHub** 탭을 고릅니다.
3. 칸에 `AI-Engineering-Teamproject/Lite-R-CNN`을 넣고 엔터. 저장소가 비공개면 **비공개 저장소 포함**(Include private repos)에 체크하고 GitHub 연결을 허용합니다.
4. **브랜치**에서 `perf/improve`를 고르고 `notebooks/Lite_RCNN_T4_Colab.ipynb`를 누릅니다.

GitHub 탭이 안 되면: GitHub 웹에서 이 파일을 열어 **Download raw file**(⬇)로 받고, Colab에서 **파일 → 노트북 업로드**로 올립니다.

### 실행

1. **런타임 → 런타임 유형 변경 → 하드웨어 가속기: T4 GPU → 저장** (노트북에 T4가 저장되어 있어 보통 이미 선택되어 있습니다).
2. **1. 설정값**은 그대로 둡니다. 브랜치 이름을 바꿨다면 `BRANCH`만 고칩니다.
3. **런타임 → 모두 실행**(Run all, `Ctrl+F9`).
4. 창이 뜨면:
   - "이 노트북은 Google에서 작성하지 않았습니다" → **무시하고 계속하기**(Run anyway)
   - Google Drive 액세스 허용 → **Google Drive에 연결 → 계정 선택 → 허용**. 결과를 드라이브에도 백업해 세션이 끊겨도 남깁니다.
     원하지 않으면 설정에서 `SAVE_TO_DRIVE`를 끄고 실행하세요.
   - 두 창에 답해야 다음으로 넘어가므로 **처음 1~2분은 자리를 지키세요.** 설치 중 빨간 `dependency conflicts` 경고는 무시해도 됩니다.
5. 셀마다 아래에 출력이 계속 나옵니다. 단계 B는 모델마다 `[9/18] Light R-CNN v2 · 빠른 설정 (fast)`처럼 설정별 정확도를 먼저 잰 뒤,
   `[지연 시간] … 교차 측정` 줄과 함께 그 모델의 설정을 번갈아 가며 지연 시간을 한꺼번에 잽니다.
6. **실행 중에는 브라우저 탭을 닫지 말고, 컴퓨터가 잠자기에 들어가지 않게** 하세요 (Windows: 설정 → 시스템 → 전원 → 절전 '안 함').

| 단계 | 하는 일 | T4 예상 시간 |
|---|---|---|
| 0~3 | GPU 확인, 저장소 받기·설치, COCO val2017 내려받기 (약 1 GB) | 5~10분 |
| A | 공정한 기준 비교 (`scripts/run_comparison.py`): Faster R-CNN · v2 · YOLO26, 이어서 v3 | 20~35분 |
| B | 개선 전/후 비교 (`scripts/compare_improvements.py`): 추론 설정 6가지 × 3개 모델 | 70~100분 |
| C (선택, 기본 꺼짐) | 팀 세 모델 평가 (`scripts/evaluate_three_models_gpu_full.py`) | 20~30분 |

예상 시간은 T4에서 직접 잰 값이 아니라 추정치입니다. 처음이라 불안하면 설정에서 `QUICK_TEST`를 켜고
15~20분짜리 점검(이미지 50장, 짧은 측정)을 먼저 돌린 뒤, 끄고 다시 **모두 실행**하세요. 다만 무료 GPU 시간을 그만큼 더 씁니다.
점검 결과는 `results_quick.zip`에 따로 담기고 본 측정 `results.zip`에는 섞이지 않습니다.

### 결과 보내기

끝나면 `results.zip`이 자동으로 내려받아집니다. 안 되면 왼쪽 📁 아이콘 → `results.zip` 오른쪽 클릭 → **다운로드**.
드라이브 저장을 켰다면 `내 드라이브/Lite-R-CNN-results/results_<실행 시각>.zip`에도 있습니다 (단계가 끝날 때마다 갱신).
오래 걸리는 단계 B는 모델 하나(그 모델의 추론 설정 전부)가 끝날 때마다 `step_b_<시각>/` 폴더에도 복사합니다.

보낼 것:

1. `results.zip`. 중간에 세션이 끊겨 다시 돌렸다면 새 `results.zip`에는 끊기기 전에 끝난 결과가 없으므로, `내 드라이브/Lite-R-CNN-results`의
   `results_<시각>.zip`을 **모두**, `step_b_<시각>` 폴더가 있으면 그것도 보냅니다 (폴더는 드라이브에서 오른쪽 클릭 → 다운로드하면 zip으로 받아집니다.
   이름에 `quick`이 들어간 것은 빠른 점검이라 빼도 됩니다).
2. 오류가 있었다면 **파일 → 다운로드 → .ipynb 다운로드**로 받은 노트북 (출력이 함께 저장되어 원인을 볼 수 있습니다)
3. 한 줄 메모: 어떤 단계를 돌렸는지, 중간에 끊겼는지

`results.zip`에는 같은 런타임에서 끝낸 단계의 결과가 들어 있습니다. 빠른 점검 ↔ 본 측정을 바꾸면 그때부터 새로 담으므로
빠른 점검 결과는 섞이지 않습니다 (빠른 점검은 `results_quick.zip`에 따로).

```text
outputs/comparison_runs/<시각>/         단계 A (v2 실행, v3 실행 각각 한 폴더): 정확도·지연 시간·단계별 시간 JSON, figures/report.html
outputs/improvements/colab_<시각>/      단계 B: summary.md, summary.csv, figures/, reports/v2, reports/v3
outputs/evaluation_runs/colab_<시각>/   단계 C: REPORT.md, summary.csv, evaluation.json
outputs/colab_logs/<시각>/              실행마다: 단계별 로그, GPU 정보, 설정값과 걸린 시간 (colab_run_<시각>.json)
```

### 결과 보기

1절에서 `perf/improve` 브랜치로 옮긴 저장소 폴더에서 봅니다 (`main`에는 개선 전/후 화면이 없음). 명령 프롬프트에서:

```bat
cd /d C:\Users\82103\Desktop\Lite-R-CNN
tar -xf C:\Users\82103\Downloads\results.zip
dir outputs\improvements
streamlit run dashboard/app.py
```

1. `tar -xf …`: `results.zip`을 저장소 폴더에 풉니다 (Windows 10 이상에 들어 있는 명령). 파일이 `outputs\...` 아래에 들어갑니다.
   탐색기로 풀어도 되지만, 풀 위치를 저장소 폴더로 바꿔 `results` 같은 폴더가 한 단계 더 생기지 않게 하세요.
2. `dir outputs\improvements`: 단계 B 결과 폴더 이름(`colab_<시각>`)을 확인합니다.
3. `streamlit run dashboard/app.py`: 브라우저에 대시보드가 열립니다. 가상환경(`.venv` 등)을 쓰면 그 전에 `.venv\Scripts\activate`로 켭니다
   (README의 `Activate.ps1`은 PowerShell용). 대시보드 패키지가 없으면 한 번 `python -m pip install "streamlit>=1.56" "plotly>=5.24" "pandas>=2.2" pyyaml`을
   실행합니다 (따옴표는 꼭 넣으세요. 빼면 명령 프롬프트가 `>`를 파일 저장 기호로 읽습니다).

대시보드에서:

- 사이드바 **결과 폴더**에 `outputs\improvements\colab_<시각>\reports`를 넣습니다 (`<시각>`은 2번에서 확인한 이름).
- 이 폴더에는 v2와 v3가 함께 있으므로 사이드바 **Light R-CNN 체크포인트**에서 버전을 고르고, **추론 설정**에서 `fast`, `accurate` 등을 고릅니다.
- **개선 전/후** 탭에서 기본 설정(`baseline`)과 비교합니다. p50 변화에 붙은 **95% CI**가 0을 포함하면 '변화 없음 (오차 범위)'으로
  판정됩니다 (`summary.md`의 `Δp50 (%) [95% CI]` 열과 같은 값). 이 구간은 한 번의 실행 안의 흔들림만 담습니다.
  읽는 법은 [성능 개선 계획 5절](PERFORMANCE_PLAN.md#5-결과-읽는-법)에 있습니다.
- 단계 A 결과는 **결과 폴더**에 `outputs\comparison_runs\<시각>`을 넣어 봅니다. 표만 보려면 `outputs\improvements\colab_<시각>\summary.md`를 엽니다.

### 문제 해결

- **저장소 받기 실패 (비공개 저장소)**: GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** →
  Generate new token. Resource owner는 `AI-Engineering-Teamproject`, Repository access는 `Lite-R-CNN`만, Permissions는 **Contents: Read-only**.
  조직이 토큰 승인을 요구하면 관리자 승인이 필요합니다. 만든 토큰을 Colab 왼쪽 **🔑(보안 비밀)** → 새 보안 비밀 추가 →
  이름 `GITHUB_TOKEN`, 값 붙여넣기 → **노트북 액세스** 켜기. 그다음 2번 셀부터 다시 실행합니다. 토큰은 노트북·결과에 저장되지 않습니다.
- **GPU에 연결할 수 없음 / 사용량 한도**: 무료 Colab은 하루 GPU 시간이 정해져 있습니다. 나중에 다시 하거나, 끝난 단계만 먼저 보내 주세요.
- **세션이 끊김**: 다시 연결하면 새 컴퓨터가 배정되어 `/content`의 파일이 사라집니다. 드라이브에는 끝난 단계가 담긴 `results_<시각>.zip`과,
  단계 B 도중이었다면 끝난 모델까지 복사된 `step_b_<시각>` 폴더가 남습니다. 끝난 단계는 설정에서 끄고 다시 **모두 실행**하세요
  (설치와 다운로드는 다시 합니다). 단계 B는 이어서 잴 수 없어 처음부터 다시 잽니다. `step_b_<시각>/summary.md`에 들어 있는 모델은
  모든 설정이 끝난 것이므로 `STEP_B_MODELS`에서 빼도 됩니다. 보낼 때는 위 '보낼 것' 1번처럼 드라이브의 zip과 폴더를 모두 보냅니다.
- **한 단계만 다시**: 설정에서 나머지 단계를 끄고 **모두 실행**. 같은 세션이면 저장소·COCO가 이미 있어 0~3번은 금방 끝납니다.
- **CUDA out of memory**: **런타임 → 세션 다시 시작** 후 처음부터 실행합니다.
- **COCO 내려받기 실패**: 공식 서버가 안 되면 Hugging Face 사본(`scripts/prepare_hf_coco.py`)으로 자동으로 다시 시도합니다. 둘 다 실패하면 출력 메시지를 보내 주세요.
- **단계 B에서 `gpu_graph`만 실패**: GPU에서 처음 검증하는 설정이라 그럴 수 있습니다. 나머지 결과와 함께 그대로 보내 주세요.

## 3. 학교 Linux RTX 4090에서 학습하기

학교 PC의 기존 저장소 폴더(COCO 데이터와 torch 2.7.1+cu128 가상환경이 있는 곳, 예: `~/Desktop/Lite-R-CNN`)는 **건드리지 않고**,
`git worktree`로 `perf/improve` 브랜치를 **별도 폴더**에 꺼내서 학습합니다. 데이터와 가상환경은 그대로 함께 씁니다.

### 처음 한 번: 작업 폴더 만들기

```bash
# 0) 경로 (자기 환경에 맞게 바꾸기)
MAIN=~/Desktop/Lite-R-CNN         # 기존 저장소: data/coco, .venv가 있는 곳
WT=~/Desktop/Lite-R-CNN-perf      # 새 작업 폴더

# 1) 브랜치를 별도 폴더로 꺼내기 (기존 폴더의 파일은 바뀌지 않음)
cd "$MAIN"
git fetch origin
git worktree add -b perf/improve "$WT" origin/perf/improve
#   "already exists"라고 나오면: git worktree add "$WT" perf/improve && git -C "$WT" pull

# 2) 데이터는 링크로 공유 (복사하지 않음, -sfn: 다시 실행해도 기존 data 폴더 안에 링크가 생기지 않음)
ln -sfn "$MAIN/data" "$WT/data"

# 3) 새 터미널마다 쓸 환경 설정 파일 (가상환경 이름이 .venv가 아니면 고치기)
cat > ~/lite-rcnn-perf-env.sh <<EOF
source "$MAIN/.venv/bin/activate"
export PYTHONPATH="$WT/src"
cd "$WT"
EOF
source ~/lite-rcnn-perf-env.sh

# 4) 확인: 마지막 경로가 새 폴더($WT/src/...)여야 합니다
python -c "import torch, light_rcnn; print(torch.__version__, torch.cuda.is_available(), light_rcnn.__file__)"
nvidia-smi                   # 다른 학습이 GPU를 쓰고 있지 않은지 확인
```

`PYTHONPATH`가 중요합니다. 기존 가상환경의 `pip install -e .`는 **기존 폴더의 코드**를 가리키므로, 이것 없이 실행하면
새 설정 항목(`rpn_anchor_strides`, `train.ema` 등)을 옛 코드가 읽어 실험이 조용히 틀어질 수 있습니다.
4)의 출력 경로가 `$WT/src/light_rcnn/__init__.py`인지 꼭 확인하십시오.

### 짧은 점검 (1~2분)

```bash
source ~/lite-rcnn-perf-env.sh
python scripts/train.py --config configs/coco_v3_1a.yaml --device cuda --epochs 1 \
  --train-limit 64 --val-limit 32 --output-dir outputs/smoke_v3_1a \
  --init-backbone outputs/coco_4090_v2/best.pth
rm -rf outputs/smoke_v3_1a   # 점검용 결과는 지움
```

출력에 `Initialized … MobileNetV3 tensors`와 `Training … distillation=True; ema=False; multiscale=False` 줄이 보이고
1 epoch가 오류 없이 끝나면 됩니다 (이때 AP는 의미 없음). EMA config라면 `ema=True`로 나와야 합니다.

### 본 실험: 12 epoch ablation

순서와 채택 기준은 [성능 개선 계획 3절](PERFORMANCE_PLAN.md#3-재학습-ablation-학교-rtx-4090)에 있습니다.
하나에 약 6~7시간이므로 `tmux` 안에서 차례로 돌립니다 (SSH가 끊겨도 계속 돕니다). 두 번에 나눠 실행합니다.

1) 먼저 이 한 줄만 실행합니다. tmux 화면이 새로 열립니다.

```bash
tmux new -s abl              # 나올 때: Ctrl+B 누른 뒤 D / 다시 볼 때: tmux attach -t abl
```

2) 열린 tmux 화면 안에서 아래 블록을 붙여 넣습니다. 1)과 한 번에 붙여 넣으면 이 부분이 tmux 밖에서 돌아,
SSH가 끊길 때 학습도 멈춥니다.

```bash
source ~/lite-rcnn-perf-env.sh
mkdir -p logs
for cfg in coco_v3_short coco_v3_1a coco_v3_1a_nokd coco_v3_1a_ema coco_v3_1b; do
  python scripts/train.py --config configs/$cfg.yaml --device cuda \
    --init-backbone outputs/coco_4090_v2/best.pth > logs/$cfg.log 2>&1
done
```

다섯 개를 합치면 30~35시간쯤 걸립니다. `tmux`가 없으면 `source ~/lite-rcnn-perf-env.sh`와 `mkdir -p logs` 다음에 이렇게 실행합니다.

```bash
nohup bash -c 'for cfg in coco_v3_short coco_v3_1a coco_v3_1a_nokd coco_v3_1a_ema coco_v3_1b; do python scripts/train.py --config configs/$cfg.yaml --device cuda --init-backbone outputs/coco_4090_v2/best.pth > logs/$cfg.log 2>&1; done' > logs/queue.log 2>&1 &
```

| config | 결과 폴더 | 실험 |
|---|---|---|
| `coco_v3_short` | `outputs/abl12_v3_baseline` | S0: v3 그대로 12 epoch (기준) |
| `coco_v3_1a` | `outputs/abl12_v3_1a_anchors` | A1: 앵커 9개 + 고정 stride |
| `coco_v3_1a_nokd` | `outputs/abl12_v3_1a_nokd` | A3: A1에서 증류 끔 |
| `coco_v3_1a_ema` | `outputs/abl12_v3_1a_ema` | A1 + EMA |
| `coco_v3_1b` | `outputs/abl12_v3_1b_anchors_roi64` | A2: A1 + RoI 채널 64 |
| `coco_v3_1` | `outputs/coco_4090_v3_1` | v3.1 전체 48 epoch (ablation 결과를 본 뒤) |

증류 손실 수정(KD)과 multi-scale 실험은 config를 A1에서 만들어 씁니다 (설정 값은 검증 전 추정치).

```bash
python - <<'EOF'
import yaml

base = yaml.safe_load(open("configs/coco_v3_1a.yaml", encoding="utf-8"))
variants = {
    "coco_v3_1a_kdfix": {**base, "distillation": {**base["distillation"],
                                                  "feature_loss": "standardized", "box_weight": 1.0},
                         "train": {**base["train"], "output_dir": "outputs/abl12_v3_1a_kdfix"}},
    "coco_v3_1a_ms": {**base, "train": {**base["train"], "output_dir": "outputs/abl12_v3_1a_ms",
                                        "multiscale": {"enabled": True, "max_size": 640,
                                                       "min_sizes": [384, 416, 448, 480, 512,
                                                                     544, 576, 608, 640]}}},
}
for name, config in variants.items():
    with open(f"configs/{name}.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    print(f"configs/{name}.yaml")
EOF
```

만든 config는 위 `for` 문의 목록에 `coco_v3_1a_kdfix coco_v3_1a_ms`로 넣어 같은 방식으로 돌립니다.

### 중간에 멈췄을 때

```bash
source ~/lite-rcnn-perf-env.sh
python scripts/train.py --resume outputs/abl12_v3_1a_anchors/last.pth   # 마지막으로 끝난 epoch부터
```

설정과 총 epoch는 체크포인트에 저장된 값을 씁니다. 끝나면 `for` 문에서 남은 config만 다시 돌립니다.

### 진행 상황 보기

```bash
tail -f logs/coco_v3_1a.log                  # 학습 로그 (Ctrl+C로 보기만 멈춤, 학습은 계속)
watch -n 10 nvidia-smi                       # GPU 사용률·메모리
cat outputs/abl12_v3_1a_anchors/RESULTS.md   # epoch마다 갱신되는 요약
python - <<'EOF'
import json
import pathlib

for history in sorted(pathlib.Path("outputs").glob("abl12_*/history.jsonl")):
    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("validation")]
    if not rows:
        continue
    best = max(rows, key=lambda row: row["validation"]["AP_50_95"])
    ap = {key: 100 * value for key, value in best["validation"].items() if key.startswith("AP")}
    print(f"{history.parent.name:32s} epoch {rows[-1]['epoch']:2d} | best epoch {best['epoch']:2d}"
          f"  AP {ap['AP_50_95']:.2f}  AP75 {ap['AP_75']:.2f}  AP_S {ap['AP_small']:.2f}"
          f"  AP_L {ap['AP_large']:.2f}")
EOF
```

### 보내 줄 것

```bash
source ~/lite-rcnn-perf-env.sh
tar czf ablation_results.tgz logs outputs/abl12_*/RESULTS.md outputs/abl12_*/history.jsonl outputs/abl12_*/config.yaml
```

- `ablation_results.tgz` (작은 파일들만: 로그, 요약, epoch 기록, 설정)
- 채택 기준을 넘은 실험의 `best.pth` (실험당 약 45 MB)는 따로 요청할 때 보내 주세요.
- 학습 중 오류가 났다면 해당 `logs/<config>.log`의 마지막 50줄

브랜치에 새 커밋이 올라오면 `git -C ~/Desktop/Lite-R-CNN-perf pull`로 받습니다 (학습 중인 실험이 없을 때).

작업 폴더를 다 쓰면 결과(`logs/`, `outputs/abl12_*`, 직접 만든 `configs/coco_v3_1a_kdfix.yaml` 등)를 먼저 다른 곳에
옮겨 둔 뒤 지웁니다 (경로는 0)에서 정한 것). 데이터 링크와 남은 파일 때문에 `--force`가 필요합니다. 작업 폴더 안의
파일은 모두 지워지지만, 링크만 지울 뿐 기존 폴더의 `data/coco`는 그대로 남습니다.

```bash
git -C ~/Desktop/Lite-R-CNN worktree remove --force ~/Desktop/Lite-R-CNN-perf
```
