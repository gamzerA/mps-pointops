<!-- SPDX-License-Identifier: MIT -->
# Ball Query 수학·수치 계약

이 문서는 이식한 Metal 커널과 내부
`mps_pointops._ball_query_mps.ball_query`의 수치 계약을 설명한다. 공개
`mps_pointops.ball_query(query, ref, radius, K)`는 이를 호출해 `(S, I)`
튜플을 반환하며, CPU 입력은 `reference.ball_query`로 실행한다.

반경으로 점군의 지역 이웃을 정의하는 근거는 [PointNet++ 원 논문](https://papers.nips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)이다. 첫 구현은 [PyTorch3D 공개 API](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/ball_query.py)와 [CPU](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/ball_query/ball_query_cpu.cpp)·[CUDA](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/ball_query/ball_query.cu) 동작을 참고해 수식으로 직접 정의했다. 원문 코드를 옮긴 명세가 아니다. 아래의 실수 수식은 연산의 의미이며, 부동소수점 결과의 비트 단위 일치를 약속하지 않는다.

## 수학적 계약

배치 수를 `B`, 배치당 쿼리 수를 `Q`, 검색 대상 점 수를 `P`, 좌표 차원을 `D = 3`, 최대 이웃 수를 `K ≥ 0`으로 둔다. 쿼리 좌표는 `q ∈ ℝ^(B×Q×3)`, 대상 좌표는 `x ∈ ℝ^(B×P×3)`이다. 배치별 유효 길이는 `LQ[b] ∈ {0,…,Q}`, `LX[b] ∈ {0,…,P}`이며, 생략하면 각각 `Q`, `P`다. 모든 인덱스 `b,i,j,d,k`는 0부터 센다.

유효한 쿼리 `0 ≤ i < LQ[b]`와 점 `0 ≤ j < LX[b]`의 실수 제곱거리를

~~~text
s[b,i,j] = Σ_(d=0)^2 (q[b,i,d] − x[b,j,d])²
~~~

로 정의한다. 점 `j`가 반지름 `r` 안에 있으려면 `s[b,i,j] < r²`이어야 한다. 정확히 경계에 놓인 점과 `r = 0`에서의 동일 좌표는 제외된다. 이 엄격한 부등호는 PyTorch3D [CPU](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/ball_query/ball_query_cpu.cpp)·[CUDA](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/ball_query/ball_query.cu)의 비교와 같다. 실제 float32 판정식은 아래에 따로 명시한다.

각 `(b,i)`에서 조건을 만족하는 `j`를 **입력 인덱스 오름차순**으로 나열한 목록을 `J[b,i]`, `m[b,i] = min(K, |J[b,i]|)`라고 한다. 출력 인덱스 `I ∈ ℤ^(B×Q×K)`와 출력 제곱거리 `S ∈ ℝ^(B×Q×K)`는 `0 ≤ k < m[b,i]`에서

~~~text
I[b,i,k] = J[b,i][k]
S[b,i,k] = s[b,i,I[b,i,k]]
~~~

이다. 나머지 칸과 유효 길이 밖의 쿼리는 `I = -1`, `S = 0`으로 채운다. 이는 거리순 최근접 `K`개가 아니다. **`S = 0`은 패딩, 동일 좌표, 거리 언더플로우 어느 경우든 가능**하므로 유효성은 `I ≥ 0`으로만 판단한다. 이 인덱스·거리·패딩 의미는 [PyTorch3D API](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/ball_query.py)를 따른다.

## 현재 Python API와 입력 검증

내부 함수는 `ball_query(queries, points, *, radius, k, query_lengths=None, point_lengths=None)`이며 `BallQueryResult(distances=S, indices=I)`를 반환한다. 입력 형상은 각각 `(B,Q,3)`, `(B,P,3)`이고 같은 MPS 장치와 같은 `float32` 또는 `float16` dtype이어야 한다. 다른 dtype은 변환하지 않고 거부한다. `float16` 좌표는 거리 계산 때 `float32`로 올린다. `S`는 `float32`, `I`는 `int64`이고 좌표 기울기는 입력 dtype으로 돌아온다. 비연속 입력은 MPS에서 연속 버퍼로 복사한다.

`query_lengths`와 `point_lengths`는 각각 모양 `(B,)`, dtype `int64`, 입력과 같은 MPS 장치의 텐서여야 한다. 호스트는 `0 ≤ LQ[b] ≤ Q`, `0 ≤ LX[b] ≤ P`를 검사하고 범위를 벗어나면 오류를 낸다. **명시적 lengths의 범위 검사는 `.item()`으로 GPU→CPU 동기화를 일으킨다.** 범위 밖 값을 커널에서 조용히 clamp하는 계약은 아니다. 커널의 상한 처리는 방어적이다. `k`는 음수가 아닌 Python 정수이며, `radius`는 유한한 음이 아닌 Python 숫자다. Tensor 반지름은 받지 않는다.

`B = 0`, `Q = 0`, `K = 0`이면 올바른 모양의 빈 출력 텐서를 반환하고 스레드를 디스패치하지 않는다. `P = 0`이고 출력 칸이 있으면 모든 칸을 `I = -1`, `S = 0`으로 채운다. 유효 lengths가 0인 경우도 같은 패딩 규칙을 따른다. 출력 형상과 입력 요소 수의 정수 범위는 호스트에서 검사한다.

이는 **PyTorch3D의 전체 호출 서명과 아직 호환되지 않는다.** 최신 PyTorch3D는 `p1, p2, lengths1, lengths2, K, radius, return_nn, skip_points_outside_cube`를 받으며 기본값과 선택적 이웃 좌표 `knn` 출력이 있다. `skip_points_outside_cube=True`는 반경 바깥의 축 정렬 큐브에 놓인 점을 거리 계산 전에 거르는 최적화 인자다. 현재 API는 두 인자를 받지 않고 이웃 좌표도 반환하지 않는다. PyTorch3D의 일반 좌표 차원과 달리 이 Metal 커널은 `D = 3`으로 제한된다. [PyTorch3D 공개 API](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/ball_query.py)

## 커널 구현과 조기 종료

Metal 커널은 각 `(b,i)`에 SIMD 그룹 하나를 배정한다. 그룹은 `j = 32t, …, 32t+31` 블록을 `t = 0,1,…` 순서로 검사하고, 마지막 블록에서 범위 밖 lane은 불일치로 취급한다. 블록 `t`의 lane `ℓ`이 일치하면 `h_{t,ℓ}=1`, 아니면 0이다. 그룹의 결정적 배타 접두 합 `r_{t,ℓ}=Σ_{u<ℓ}h_{t,u}`가 블록 안에서의 순위이고, `H_t=Σ_ℓ h_{t,ℓ}`가 블록의 일치 개수다. 앞선 블록에서 이미 기록한 개수를 `F_t`라 하면, 일치 lane은 `F_t+r_{t,ℓ}<K`일 때만 그 번호의 출력 칸에 기록한다. 블록 뒤에는 `F_{t+1}=min(K,F_t+H_t)`로 갱신한다.

블록마다 이 규칙을 적용하면 처리한 접두 구간의 출력은 항상 원래 `j` 순서에서 첫 `min(K, 일치 개수)`개라는 불변식이 유지된다. 각 출력 칸의 순위가 유일하므로 atomic 기록 경쟁도 없다. `F_{t+1}=K`가 되면 이후 블록에는 앞선 K개보다 작은 `j`가 없으므로 조기 종료해도 결과가 같다. 병렬성은 쿼리 방향과 각 블록의 32개 `j` 방향에 있다. 이 논증은 SIMD 접두 순위와 블록 순서에 의존한다. kNN처럼 방문 블록을 섞거나 일치 lane이 경쟁적으로 슬롯을 차지하는 구현에는 적용되지 않는다.

호스트는 `(B,Q,K)`의 `int64` 인덱스와 `float32` 제곱거리 버퍼를 `torch.empty`로 할당한다. 커널에서 일치 항목은 위의 서로 다른 접두 순위 슬롯에 쓰고, `F=found`에서 끝나면 각 lane `ℓ`이 `F+ℓ, F+ℓ+32, … < K`의 남은 슬롯을 `(-1, 0)`으로 채운다. 따라서 유효 항목과 패딩을 합쳐 **모든 출력 슬롯에 기록**된다. 유효 쿼리가 없거나 `P=0`이어도 패딩 루프가 실행된다. `B·Q·K=0`이면 버퍼가 비어 있으므로 디스패치하지 않는다. 센티널로 미리 채운 출력 버퍼에 커널을 직접 실행해 이 전 범위 기록과 공개 API 반환 dtype을 시험한다.

Metal 호출은 PyTorch의 현재 MPS 스트림에 제출되므로 Python 함수가 돌아올 때 GPU 완료를 뜻하지 않는다. 동일 스트림의 후속 PyTorch 연산은 순서대로 실행된다. 벤치마크는 측정 직전·직후 [`torch.mps.synchronize()`](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html)를 호출해 비동기 실행 시간을 포함한다. 별도 Metal 큐와 버퍼를 공유하는 응용은 자체 동기화가 필요하다.

최악 시간은 `O(B·Q·P)`, 출력 저장 공간은 `O(B·Q·K)`다. 전체 `B×Q×P` 거리 행렬을 만들지 않는다. 실제 검사 수는 lengths와 조기 종료 시점에 좌우된다.

## 부동소수점 판정

`fl32(z)`를 실수 `z`를 IEEE 754 float32로 최근접 짝수 반올림한 값이라고 쓰자. Python 반지름 `r`을 먼저 `r̂ = fl32(r)`로 바꾸고, 임계값을 **`R₂ = fl32(r̂ · r̂)`**로 만든다. 이는 [PyTorch3D CPU](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/ball_query/ball_query_cpu.cpp)와 [CUDA](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/ball_query/ball_query.cu)의 `float radius; const float radius2 = radius * radius;`와 같은 반올림 순서다. 호스트 코드의 float64 곱은 두 float32 유효숫자의 정확한 곱을 표현할 수 있으므로, 뒤의 float32 반올림 한 번으로 `R₂`를 얻는다.

예를 들어 Python `r = 0.1`이면 `R₂ = 0.010000000707805157` (`0x3c23d70b`)이다. **원래 Python float64 값 `0.1`을 먼저 제곱하고 float32로 바꾼 값**은 `0.009999999776482582` (`0x3c23d70a`)로, 한 ULP 작다. 이 둘을 모호하게 “반올림된 r²”로 합치지 않는다.

실수에서 `r > 0`이면 `Σ_d(q_d−x_d)² < r² ⇔ Σ_d((q_d−x_d)/r)² < 1`이다. float32에서는 뺄셈·제곱·나눗셈·합산의 반올림과 서브노멀 처리 때문에 두 판정이 일반적으로 같지 않다. 구현은 **float32로 반올림한 양의 `r̂ < 2^-50`이거나 `R₂ = +∞`이면** 성분 차이를 `r̂`로 나눈 정규화 합 `T`를 1과 비교하고, 그 밖에는 직접 제곱거리 합을 `R₂`와 비교한다. 이 분기점은 PyTorch3D와 비트 일치를 보증하는 정리가 아니라 작은 중간 곱의 FTZ 위험을 줄이기 위한 정책이다. 정규화 경로에서 매치된 점의 출력 `S`는 `R₂`가 유한한 정규수면 `fl32(T·R₂)`로 복원한다. `R₂`가 0·서브노멀·무한대라면 `Σ_d(q_d−x_d)²`를 직접 계산해 내보낸다. 후자의 출력은 0 또는 `+∞`가 될 수 있다. 이 출력 복원도 float32 수치 정책이며 PyTorch3D의 직접 거리 계산과 동일한 비트 패턴을 보장하지 않는다.

직접 경로에서 `R₂`가 정규수여도 개별 `(q_d−x_d)²`가 서브노멀이면 성분이 사라질 수 있다. 예를 들어 실수 `r = 2^-62`, `q−x = (2^-64, 0.98·2^-62, 0)`이면 `s/r² = 1/16 + 0.98² = 1.0229 > 1`이다. 작은 성분의 제곱 `2^-128`이 사라지면 `0.9604 < 1`이 되어 잘못 포함된다. M5 Pro Safe probe에서 `(작은 x, 큰 y, 0)`과 `(0, 작은 y, 큰 z)`의 직접 합은 `4.5157793093e-38 < R₂ = 4.7019774033e-38`였고, 정규화 합은 `1.0228999853 > 1`이었다. 첫 배치에서는 `x·x` 결과가 서브노멀이다. 둘째 배치에서는 **`fma(y,y,0)`의 최종 결과 자체가 서브노멀**이므로, 곱셈이 FMA 안에 있어도 결과가 flush될 수 있다. 반대로 `(큰 x, 0, 작은 z)`와 추가로 측정한 `(큰 x, 작은 y, 0)`의 직접 합은 모두 `4.8096531773e-38 > R₂`였다. 이때 작은 성분을 더하기 전의 누산값은 정규수여서 FMA의 최종 결과도 정규수다. 따라서 처음 세 축 배치의 관찰만으로 두 번째 FMA가 곱셈·덧셈으로 분리되었다고 추론할 수 없다. 추가 배치는 작은 y의 기여가 보존됨을 보여주지만, 기계 명령을 검사한 증거는 아니다. 이 값들은 해당 기기의 관찰이며 모든 Apple GPU의 보증은 아니다.

`2^-50 ≈ 8.88×10^-16` 기준은 중간 곱의 flush 오차가 보통의 반올림 오차보다 커지는 대략 `10^-15` 이하 영역을 포괄한다. 정규화 경로도 `q_d−x_d` 자체가 flush되면 오분류할 수 있으며, 대략 `10^-34` 이하 반지름에서 특히 주의가 필요하다. 이 두 수치는 위험 구간의 규모를 설명하는 값이지 정확성 경계가 아니다. 양의 반지름은 **반올림 후 `r̂ ≥ 2^-112`**여야 하며 그 아래는 거부한다. `r = 0`은 허용하지만 매치가 없다. 하한 위에서도 극단 좌표의 비트 단위 PyTorch3D 호환은 보장하지 않는다.

기존 `r = 1e-40` probe에서는 동일 좌표의 정규화 합이 NaN이었다. 로드된 반지름 비트는 `0x000116c2`로 비영이었지만 `0/r̂ = NaN`, `1/r̂ = +∞`가 관찰됐다. 소스는 GPU에서 `delta / r̂`를 쓰므로 **호스트 역수 선계산은 하지 않는다**. 컴파일러가 나눗셈을 어떤 명령으로 내리는지까지 확인하지 않았으므로, 분모 FTZ와 내부 역수 오버플로 중 어느 쪽이 직접 원인인지 단정하지 않는다. 이 반지름은 새 API 하한 아래라서 거부된다.

## 컴파일 설정과 검증 기준

이 프로젝트는 `torch.mps.compile_shader(source)`를 사용한다. [PyTorch 2.14.0 Metal 컴파일 경로](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/mps/OperationUtils.mm#L790-L820)는 macOS 15 이상에서 `PYTORCH_MPS_FAST_MATH`가 미설정 또는 `0`이면 `MTLMathModeSafe`와 `MTLMathFloatingPointFunctionsPrecise`, 0이 아니면 `MTLMathModeFast`와 `MTLMathFloatingPointFunctionsFast`를 설정한다. macOS 26 이상은 Metal 언어 버전 4.0, macOS 15 이상은 3.2를 택한다. 이 설정은 PyTorch 프로세스의 첫 Metal 라이브러리 컴파일 때 캐시되므로 Safe와 Fast 비교는 각각 **새 프로세스**에서 실행한다.

Metal **소스 수준**에서는 `#pragma METAL fp contract(off)`를 넣고 직접 거리 합을 `d₀·d₀`, `fma(d₁,d₁,…)`, `fma(d₂,d₂,…)`로 작성했다. 암묵적 FMA contraction과 성분 합산 순서의 불확실성을 줄이려는 의도다. 위 probe는 작은 y가 정규수 누산값에 더해질 때 보존되는 동작을 관찰했지만, 컴파일된 기계 명령 자체는 확인하지 않았다. 뺄셈·나눗셈·서브노멀 처리, 컴파일러와 하드웨어 차이까지 없애지는 못하므로 PyTorch3D CPU/CUDA와의 비트 동등성은 약속하지 않는다. [Apple Metal Shading Language 명세 §1.6.3·§8.1](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)은 Safe에서도 기본 contraction을 허용하고 서브노멀 입력·결과를 0으로 flush할 수 있다고 설명한다. Fast에서는 NaN/Inf 관련 보증도 달라진다.

[Metal FP probe](../tools/metal_fp_probe.py)와 M5 Pro / macOS 26.5.2 / PyTorch 2.14.0의 [Safe 원시 결과](metal-fp-probe-safe.json)·[Fast 원시 결과](metal-fp-probe-fast.json)를 보존했다. 기존 probe에서 Safe의 기본 `a*b+c`는 FMA와 일치하는 값을 냈고 contraction을 끄면 달라졌으며, 두 모드에서 `2^-126·0.5`는 0이었다. 작은 성분을 x·y·z에 놓는 실험에 더해, 정규수 누산값 뒤에 작은 y를 놓는 배치를 별도로 측정했다. 동일 좌표의 정규화 합도 관찰했다.

기본 정확성 검사는 `PYTORCH_MPS_FAST_MATH=0`에서, Fast 모드 회귀 관찰은 별도 프로세스에서 실행한다. 정확히 표현되는 `r = 1`의 축 방향 경계와 바로 안팎에서는 인덱스의 완전 일치를 요구한다. 현재 **직접 경로 경계 테스트**는 반지름 `0.1, 0.3, 1, 3`을 float32로 반올림하고 좌표도 float32로 만든 뒤, binary64에서 제곱거리 `s64`를 계산해 `R₂`와 비교한다. `|s64 − float64(R₂)| ≤ 4·ulp32(R₂)`인 점은 경계 밴드로 분류해 어느 판정도 허용하고, 이 밴드 밖의 테스트 점은 인덱스 완전 일치를 요구한다. **정규화 경로 경계 테스트**는 `r = 2^-62`, `1e-30`, `1e20`에서 `|s64/(r̂²) − 1| ≤ 4·2^-23`을 무차원 경계 밴드로 사용하고, 그 밖의 포함·제외는 완전 일치를 요구한다. 각 반지름에서 밴드 안과 밖의 양쪽 사례가 있는지 확인한다. 두 기준은 이 테스트 입력에서 Safe·Fast 각각 확인한 수용 규칙이지 모든 입력의 오차 보증이 아니다. 작은 성분의 FTZ와 overflow는 별도 사례로 검증한다.

## 비유한 입력

Safe 모드에서 NaN 또는 ±∞가 포함된 쿼리는 이웃을 갖지 않고, 그런 대상 점은 어떤 쿼리의 이웃도 되지 않는다. 유한 좌표끼리의 차이가 float32에서 무한대가 된 경우도 후보에서 제외한다. Fast 모드에서는 비유한 값 처리를 계약으로 보증하지 않고 기기별 실측으로만 기록한다.

## 역전파

인덱스 `I`와 반지름 판정은 이산 선택이므로 미분하지 않는다. 선택된 이웃을 forward에서 고정한 뒤 `G[b,i,k] = ∂ℒ/∂S[b,i,k]`라 하자. `j = I[b,i,k] ≥ 0`인 칸에서 거리 식을 미분하면

~~~text
∂s[b,i,j]/∂q[b,i,d] = 2(q[b,i,d] − x[b,j,d])
∂s[b,i,j]/∂x[b,j,d] = 2(x[b,j,d] − q[b,i,d])

∂ℒ/∂q[b,i,d]
  = Σ_(k: I[b,i,k] ≥ 0) 2G[b,i,k](q[b,i,d] − x[b,I[b,i,k],d])

∂ℒ/∂x[b,j,d]
  = Σ_(i,k: I[b,i,k] = j) 2G[b,i,k](x[b,j,d] − q[b,i,d])
~~~

를 얻는다. `I = -1`인 칸은 합에서 빠지며 upstream 값이 NaN이어도 기울기에 섞이지 않도록 마스킹한다. 유효 길이 밖의 좌표와 비유한 좌표도 기울기가 0이다. 같은 대상 점에 모인 기여는 `scatter_add_`로 합산하므로 **그 결과의 비트 단위 결정성은 현재 보증하지 않는다**. PyTorch3D CUDA도 [`atomicAdd`를 사용하고 비결정성을 명시](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/knn/knn.cu)한다. `I` 자체의 gradient와 경계에서 선택이 바뀌는 효과는 정의하지 않는다. `radius`는 Python 숫자라 gradient API가 없다. **선택 인덱스를 고정한 유한 입력의 거리 식**에 대해서는 Python backward를 통한 2차 미분을 지원하며, MPS 회귀 테스트에서 각 축의 `∂²s/∂q² = 2`, `∂²s/∂q∂x = -2`, `∂²s/∂x² = 2`를 확인했다. 경계의 선택 변화는 미분 대상이 아니며 모든 극단값과 dtype에 대한 2차 미분 보증은 없다. PyTorch3D는 [`once_differentiable`](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/ball_query.py)을 사용한다.

이 1차 미분은 표준 연쇄법칙으로 얻은 식이며, [PyTorch3D의 `knn_points_backward` (`norm=2`)](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/csrc/knn/knn.cu)와 같다. PyTorch3D는 `-1` 인덱스를 건너뛴다. 이 식을 PointNet++ 논문의 고유 수식으로 귀속하지 않는다. 출력 거리가 0 또는 `+∞`로 반올림되는 극단값에서도 위 식은 선택된 거리의 해석적 기울기이며 반올림된 출력의 수치 미분이라는 뜻은 아니다.

## 다른 Ball Query 계열과의 차이

[PointNet++ 논문](https://papers.nips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)은 set-abstraction에서 반경 기반 grouping을 사용한다. [원 저자의 CUDA query-ball 커널](https://github.com/charlesq34/pointnet2/blob/master/tf_ops/grouping/tf_grouping_g.cu)도 입력 순서의 첫 K개를 찾고 조기 종료하므로 **그 순서는 이 프로젝트만의 차이가 아니다**. 그 커널은 인덱스·개수를 반환하고 이웃이 있을 때 첫 인덱스를 반복해 남은 칸을 채운다. [TensorFlow wrapper](https://github.com/charlesq34/pointnet2/blob/master/tf_ops/grouping/tf_grouping.py)는 query-ball 자체에 gradient가 없다고 지정하지만 PointNet++ 모델 전체의 좌표 gradient가 없다는 뜻은 아니다. 본 구현은 PyTorch3D식 `-1`/`0` 패딩, 제곱거리 반환, 좌표 역전파, 배치별 길이를 택하고 `D = 3`과 MPS로 범위를 제한한다. [MMCV Ball Query CUDA](https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/csrc/common/cuda/ball_query_cuda_kernel.cuh)는 최소·최대 반지름과 첫 이웃 인덱스 반복 패딩을 사용해 별도 계약이다.
