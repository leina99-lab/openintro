"""
제6장 그림 생성 스크립트
범주형 자료에 대한 추론의 기초 개념을 시각화한다.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
import os

# 한글 폰트 설정 (시스템에 따라 인식되는 이름)
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = '/home/claude/ch6/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), facecolor='white')
    plt.close()


# ============================================================
# 그림 6.0.1: 범주형 자료 추론의 4가지 시나리오 개요
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

scenarios = [
    ("(a) 단일 비율",
     "Q: 미국 성인의 50% 이상이\n     백신을 지지하는가?",
     "p̂ = 0.55, n = 1000",
     "검정: 1-비율 Z-검정\n또는 카이제곱 적합도",
     '#3182CE'),
    ("(b) 두 비율의 차이",
     "Q: 남녀 흡연율에\n     차이가 있는가?",
     "p̂₁ = 0.21 (남)\np̂₂ = 0.16 (여)",
     "검정: 2-비율 Z-검정\n또는 2×2 카이제곱",
     '#38A169'),
    ("(c) 적합도 검정",
     "Q: 주사위가 공정한가?\n     6면 비율이 1/6씩?",
     "관측: [12, 8, 11, 9, 10, 10]\n기대: [10, 10, ...]",
     "검정: 카이제곱 적합도\ndf = k - 1",
     '#D69E2E'),
    ("(d) 독립성 검정",
     "Q: 학력과 정치 성향은\n     독립인가?",
     "이원분할표\n(R × C)",
     "검정: 카이제곱 독립성\ndf = (R-1)(C-1)",
     '#C53030'),
]

for ax, (title, q, data, test, color) in zip(axes.flat, scenarios):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    rect = plt.Rectangle((0.3, 0.3), 9.4, 9.4, facecolor=color, alpha=0.08,
                          edgecolor=color, linewidth=2)
    ax.add_patch(rect)

    ax.text(5, 9, title, ha='center', va='center', fontsize=14, fontweight='bold', color=color)
    ax.text(5, 7.2, q, ha='center', va='center', fontsize=10.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color))
    ax.text(5, 5, data, ha='center', va='center', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7FAFC', edgecolor='gray'))
    ax.text(5, 2.5, test, ha='center', va='center', fontsize=10,
            color=color, fontweight='bold')

plt.suptitle('제6장 개요: 범주형 자료에 대한 4가지 추론 시나리오',
             fontsize=15, fontweight='bold', y=1.00)
save('fig_6_0_1_overview.png')


# ============================================================
# 그림 6.0.2: 가설검정 4단계 흐름 (PHCC 체크리스트)
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

steps = [
    (1.5, "1. Prepare", "가설 H₀, Hₐ 설정\n유의수준 α 결정\n자료 요약 (n, p̂)", '#3182CE'),
    (5, "2. Check",   "조건 확인:\n• 독립성\n• 성공-실패 (np₀ ≥ 10)", '#38A169'),
    (8.5, "3. Calculate", "표준오차 SE 계산\n검정통계량 Z 또는 X²\np-값 계산", '#D69E2E'),
    (12, "4. Conclude", "p-값 vs α 비교\n결론을 문맥에서 서술\n(절대 '채택' 금지!)", '#C53030'),
]

for x, title, content, color in steps:
    rect = plt.Rectangle((x - 1.3, 1.5), 2.6, 4, facecolor=color, alpha=0.15,
                         edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, 5, title, ha='center', va='center', fontsize=13, fontweight='bold', color=color)
    ax.text(x, 2.8, content, ha='center', va='center', fontsize=10)

# 화살표
for i in range(3):
    x_start = 1.5 + (i + 1) * 3.5 - 1.3
    ax.annotate('', xy=(x_start - 0.05, 3.5), xytext=(x_start - 1.1, 3.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

ax.text(7, 6.4, '가설검정 4단계 (PCCC 체크리스트)',
        ha='center', va='center', fontsize=15, fontweight='bold')
ax.text(7, 0.6, '※ 5장에서 배운 절차를 6장 전반에 일관되게 적용한다.',
        ha='center', va='center', fontsize=10, style='italic', color='#666')

save('fig_6_0_2_phcc_flow.png')


# ============================================================
# 그림 6.0.3: 신뢰구간 SE vs 가설검정 SE 비교
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 왼쪽: 신뢰구간
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.add_patch(plt.Rectangle((0.5, 0.5), 9, 9, facecolor='#EBF8FF',
                            edgecolor='#3182CE', linewidth=2.5))
ax.text(5, 9, '신뢰구간', ha='center', fontsize=15, fontweight='bold', color='#2C5282')
ax.text(5, 7.8, '"모수 p는 어디쯤?"', ha='center', fontsize=12, style='italic')

ax.text(5, 6.2, '점추정 사용 (p̂)', ha='center', fontsize=12, fontweight='bold')
ax.text(5, 5,
        r'$SE = \sqrt{\dfrac{\hat{p}(1-\hat{p})}{n}}$',
        ha='center', fontsize=15)

ax.text(5, 3.2, '이유:', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 2.3,
        'p를 모르므로 가장 좋은 추정값 p̂을 대입.\n구간은 p̂ 주변에서 만들어진다.',
        ha='center', fontsize=10)

# 오른쪽: 가설검정
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.add_patch(plt.Rectangle((0.5, 0.5), 9, 9, facecolor='#FFF5F5',
                            edgecolor='#C53030', linewidth=2.5))
ax.text(5, 9, '가설검정', ha='center', fontsize=15, fontweight='bold', color='#742A2A')
ax.text(5, 7.8, '"H₀가 참이라면 이만큼 극단?"', ha='center', fontsize=12, style='italic')

ax.text(5, 6.2, '귀무값 사용 (p₀)', ha='center', fontsize=12, fontweight='bold')
ax.text(5, 5,
        r'$SE = \sqrt{\dfrac{p_0(1-p_0)}{n}}$',
        ha='center', fontsize=15)

ax.text(5, 3.2, '이유:', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 2.3,
        '"H₀가 참인 세계"에서 분포를 그려야\n하므로 p = p₀로 가정하고 계산한다.',
        ha='center', fontsize=10)

plt.suptitle('단일 비율: 신뢰구간 vs 가설검정의 표준오차',
             fontsize=14, fontweight='bold', y=1.02)
save('fig_6_0_3_se_comparison.png')


# ============================================================
# 그림 6.0.4: 두 비율의 차이 - 합동 vs 비합동 SE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 왼쪽: 신뢰구간 (비합동)
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.add_patch(plt.Rectangle((0.5, 0.5), 9, 9, facecolor='#EBF8FF',
                            edgecolor='#3182CE', linewidth=2.5))
ax.text(5, 9, '신뢰구간 (비합동 SE)', ha='center', fontsize=14, fontweight='bold', color='#2C5282')
ax.text(5, 7.5, r'$SE = \sqrt{\dfrac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \dfrac{\hat{p}_2(1-\hat{p}_2)}{n_2}}$',
        ha='center', fontsize=13)
ax.text(5, 4.5, '각 그룹의 표본비율 p̂₁, p̂₂를\n각각 사용한다.', ha='center', fontsize=11)
ax.text(5, 2.5, '이유: 신뢰구간은 두 비율이\n같다고 가정하지 않는다.',
        ha='center', fontsize=10, color='#666', style='italic')

# 오른쪽: 가설검정 (합동)
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.add_patch(plt.Rectangle((0.5, 0.5), 9, 9, facecolor='#FFF5F5',
                            edgecolor='#C53030', linewidth=2.5))
ax.text(5, 9, '가설검정 (합동 SE)', ha='center', fontsize=14, fontweight='bold', color='#742A2A')
ax.text(5, 7.5, r'$SE = \sqrt{\hat{p}_{\mathrm{pool}}(1-\hat{p}_{\mathrm{pool}})\left(\dfrac{1}{n_1}+\dfrac{1}{n_2}\right)}$',
        ha='center', fontsize=12)
ax.text(5, 5.5, r'$\hat{p}_{\mathrm{pool}} = \dfrac{x_1+x_2}{n_1+n_2}$',
        ha='center', fontsize=13)
ax.text(5, 3.5, 'H₀: p₁ = p₂ 가정 하에서\n두 자료를 합쳐 비율 추정.',
        ha='center', fontsize=11)
ax.text(5, 1.7, '이유: H₀에서 두 비율이 같으므로\n자료를 합쳐 추정하는 것이 효율적.',
        ha='center', fontsize=10, color='#666', style='italic')

plt.suptitle('두 비율의 차이: 신뢰구간 vs 가설검정의 표준오차',
             fontsize=14, fontweight='bold', y=1.02)
save('fig_6_0_4_pooled_vs_unpooled.png')


# ============================================================
# 그림 6.0.5: 카이제곱 분포의 형태
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
x = np.linspace(0, 20, 500)
dfs = [1, 2, 3, 5, 9]
colors = ['#3182CE', '#38A169', '#D69E2E', '#C53030', '#805AD5']

for df, color in zip(dfs, colors):
    y = stats.chi2.pdf(x, df)
    ax.plot(x, y, label=f'df = {df}', color=color, linewidth=2.2)

ax.set_xlabel('X² 값', fontsize=12)
ax.set_ylabel('밀도', fontsize=12)
ax.set_title('카이제곱 분포: 자유도(df)에 따른 형태 변화', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(0, 20)
ax.set_ylim(0, 0.5)
ax.grid(alpha=0.3)

# 주석 추가
ax.annotate('자유도가 커질수록\n오른쪽으로 이동하고\n분포가 평평해진다',
            xy=(12, 0.1), xytext=(15, 0.3),
            fontsize=10.5, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='gray'))

save('fig_6_0_5_chi2_distributions.png')


# ============================================================
# 그림 6.0.6: 카이제곱 분포의 직관 - 제곱들의 합
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 왼쪽: 표준정규 Z
ax = axes[0]
x = np.linspace(-4, 4, 400)
ax.plot(x, stats.norm.pdf(x), color='#3182CE', linewidth=2)
ax.fill_between(x, 0, stats.norm.pdf(x), alpha=0.3, color='#3182CE')
ax.set_title('표준정규분포 Z ~ N(0,1)', fontsize=13, fontweight='bold')
ax.set_xlabel('Z')
ax.set_ylabel('밀도')
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(alpha=0.3)
ax.text(0, 0.5, '음/양의 부호 모두 가능',
        ha='center', fontsize=10, color='#2C5282', fontweight='bold')

# 오른쪽: Z² (= chi-squared df=1)
ax = axes[1]
x = np.linspace(0.01, 10, 400)
ax.plot(x, stats.chi2.pdf(x, 1), color='#C53030', linewidth=2)
ax.fill_between(x, 0, stats.chi2.pdf(x, 1), alpha=0.3, color='#C53030')
ax.set_title('Z²의 분포 = 카이제곱(df=1)', fontsize=13, fontweight='bold')
ax.set_xlabel('Z² (= X²)')
ax.set_ylabel('밀도')
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)
ax.set_xlim(0, 10)
ax.text(5, 0.5, '항상 0 이상\n(제곱이므로)',
        ha='center', fontsize=10, color='#742A2A', fontweight='bold')

plt.suptitle('카이제곱 분포의 기원: Z² = X²(df=1)',
             fontsize=14, fontweight='bold', y=1.02)
save('fig_6_0_6_chi2_origin.png')


# ============================================================
# 그림 6.0.7: 관측값 vs 기댓값 - 카이제곱의 직관
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 왼쪽: 잘 맞는 경우
ax = axes[0]
categories = ['1', '2', '3', '4', '5', '6']
observed = [16, 18, 17, 16, 17, 16]
expected = [16.67] * 6

x_pos = np.arange(len(categories))
width = 0.38
ax.bar(x_pos - width/2, observed, width, label='관측 O',
       color='#3182CE', alpha=0.85)
ax.bar(x_pos + width/2, expected, width, label='기대 E',
       color='#E2E8F0', edgecolor='#4A5568', linewidth=1.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.set_xlabel('주사위 눈')
ax.set_ylabel('빈도')
ax.set_title('잘 맞는 경우: 작은 X²', fontsize=13, fontweight='bold', color='#2C5282')
ax.legend()
ax.set_ylim(0, 25)

# X² 계산
chi2_val_good = sum((o - e)**2 / e for o, e in zip(observed, expected))
ax.text(2.5, 22, f'X² = {chi2_val_good:.2f}\n→ H₀ 기각 안 함',
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#E6FFFA', edgecolor='#319795'))

# 오른쪽: 잘 안 맞는 경우
ax = axes[1]
observed_bad = [5, 8, 10, 15, 22, 40]
expected_bad = [16.67] * 6

ax.bar(x_pos - width/2, observed_bad, width, label='관측 O',
       color='#C53030', alpha=0.85)
ax.bar(x_pos + width/2, expected_bad, width, label='기대 E',
       color='#E2E8F0', edgecolor='#4A5568', linewidth=1.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.set_xlabel('주사위 눈')
ax.set_ylabel('빈도')
ax.set_title('잘 안 맞는 경우: 큰 X²', fontsize=13, fontweight='bold', color='#742A2A')
ax.legend()
ax.set_ylim(0, 50)

chi2_val_bad = sum((o - e)**2 / e for o, e in zip(observed_bad, expected_bad))
ax.text(1.5, 42, f'X² = {chi2_val_bad:.2f}\n→ H₀ 기각',
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#FFF5F5', edgecolor='#C53030'))

plt.suptitle('카이제곱의 직관: 관측-기대 차이가 누적될수록 X²이 커진다',
             fontsize=14, fontweight='bold', y=1.02)
save('fig_6_0_7_observed_vs_expected.png')


# ============================================================
# 그림 6.0.8: 카이제곱 검정의 결정 규칙 (오른쪽 꼬리)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))

df = 5
x = np.linspace(0, 25, 500)
y = stats.chi2.pdf(x, df)

ax.plot(x, y, color='#2C5282', linewidth=2.2)

# 임계값
crit = stats.chi2.ppf(0.95, df)
ax.fill_between(x[x >= crit], 0, y[x >= crit], alpha=0.5, color='#C53030',
                label=f'기각역 (α=0.05): X² > {crit:.2f}')
ax.fill_between(x[x < crit], 0, y[x < crit], alpha=0.2, color='#3182CE',
                label='H₀ 기각 안 함 영역')

ax.axvline(crit, color='#C53030', linestyle='--', linewidth=1.5)

# 관측 X²이 있다고 가정
obs_x2 = 12
ax.axvline(obs_x2, color='black', linewidth=2)
ax.annotate(f'관측 X² = {obs_x2}\n(기각역에 속함\n→ H₀ 기각)',
            xy=(obs_x2, 0.025), xytext=(17, 0.1),
            fontsize=10.5, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black'))

ax.set_xlabel('X² 검정통계량', fontsize=12)
ax.set_ylabel('밀도', fontsize=12)
ax.set_title(f'카이제곱 검정 (df={df}): 오른쪽 꼬리만 사용한다',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10.5)
ax.set_xlim(0, 25)
ax.grid(alpha=0.3)

ax.text(8, 0.13,
        '※ 카이제곱은 항상 단측(오른쪽) 검정.\n   X²이 클수록 관측-기대의 괴리가 크다.',
        ha='center', fontsize=10, color='#742A2A',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#742A2A'))

save('fig_6_0_8_chi2_decision.png')


# ============================================================
# 그림 6.1: 급여담보대출 가설검정 정규분포 시각화
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))

x = np.linspace(-4, 4, 500)
y = stats.norm.pdf(x)

ax.plot(x, y, color='#2C5282', linewidth=2)
ax.fill_between(x, 0, y, alpha=0.1, color='#3182CE')

# Z = 0.59
z_obs = 0.59
ax.fill_between(x[x >= z_obs], 0, y[x >= z_obs], alpha=0.6, color='#C53030')
ax.fill_between(x[x <= -z_obs], 0, y[x <= -z_obs], alpha=0.6, color='#C53030')

ax.axvline(z_obs, color='black', linewidth=1.5, linestyle='--')
ax.axvline(-z_obs, color='black', linewidth=1.5, linestyle='--')

ax.annotate(f'Z = {z_obs}\n(p̂ = 0.51)', xy=(z_obs, 0.01), xytext=(2.5, 0.15),
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFFFCC'),
            arrowprops=dict(arrowstyle='->'))

ax.text(0, 0.2, 'p-값 = 0.5552\n(양측 꼬리 면적)\n→ H₀ 기각 안 함',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

ax.set_xlabel('Z (귀무분포에서 표준화된 값)', fontsize=12)
ax.set_ylabel('밀도', fontsize=12)
ax.set_title('예제 6.6: 급여담보대출 규제 지지율 양측검정 (n=826, p̂=0.51, p₀=0.50)',
             fontsize=12.5, fontweight='bold')
ax.set_xlim(-4, 4)
ax.grid(alpha=0.3)

save('fig_6_1_payday_loan.png')


# ============================================================
# 그림 6.2: p(1-p) 함수
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
p = np.linspace(0, 1, 200)
y = p * (1 - p)

ax.plot(p, y, color='#2C5282', linewidth=2.5)
ax.fill_between(p, 0, y, alpha=0.2, color='#3182CE')

# 최대값
ax.axvline(0.5, color='#C53030', linestyle='--', linewidth=1.5)
ax.axhline(0.25, color='#C53030', linestyle='--', linewidth=1.5)
ax.scatter([0.5], [0.25], color='#C53030', s=100, zorder=5)
ax.annotate('최댓값\np = 0.5, p(1-p) = 0.25', xy=(0.5, 0.25), xytext=(0.7, 0.22),
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFFFCC'),
            arrowprops=dict(arrowstyle='->'))

ax.set_xlabel('p (모집단비율)', fontsize=12)
ax.set_ylabel('p(1-p)', fontsize=12)
ax.set_title('p(1-p) 함수: p=0.5에서 최대 (표본크기 산정의 보수적 기준)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.27)

save('fig_6_2_p_one_minus_p.png')


# ============================================================
# 그림 6.3: CPR 신뢰구간
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.5))

ci_lower, ci_upper = -0.026, 0.286
point = 0.13

# 신뢰구간 그리기
ax.plot([ci_lower, ci_upper], [1, 1], color='#2C5282', linewidth=3)
ax.scatter([ci_lower, ci_upper], [1, 1], color='#2C5282', s=80, zorder=5)
ax.scatter([point], [1], color='#C53030', s=150, marker='D', zorder=5, label=f'점추정 {point}')

# 0 선
ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(0.005, 1.4, 'p₁ - p₂ = 0\n(차이 없음)', fontsize=10, color='red')

# 경계 표시
ax.text(ci_lower, 0.8, f'{ci_lower}', ha='center', fontsize=10)
ax.text(ci_upper, 0.8, f'{ci_upper}', ha='center', fontsize=10)

ax.set_xlim(-0.15, 0.4)
ax.set_ylim(0.5, 1.8)
ax.set_yticks([])
ax.set_xlabel('생존율의 차이 (p̂_T - p̂_C)', fontsize=12)
ax.set_title('CPR 연구: 90% 신뢰구간이 0을 포함 → 차이 결론 불가',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.3, axis='x')

save('fig_6_3_cpr_ci.png')


# ============================================================
# 그림 6.4: S&P500 적합도 검정 시각화
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))

waiting = list(range(1, 8))
observed = [717, 369, 155, 69, 28, 14, 10]
expected = [743, 338, 154, 70, 32, 15, 12]

x_pos = np.arange(len(waiting))
width = 0.38
ax.bar(x_pos - width/2, observed, width, label='관측 빈도 O',
       color='#3182CE', alpha=0.85, edgecolor='black')
ax.bar(x_pos + width/2, expected, width, label='기대 빈도 E (기하분포)',
       color='#A0AEC0', alpha=0.85, edgecolor='black')

# 값 표시
for i, (o, e) in enumerate(zip(observed, expected)):
    ax.text(i - width/2, o + 15, str(o), ha='center', fontsize=9)
    ax.text(i + width/2, e + 15, str(e), ha='center', fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels([f'{w}일' for w in waiting])
ax.set_xlabel('상승 대기 일수')
ax.set_ylabel('빈도')
ax.set_title('S&P500 적합도 검정: X² = 4.61, df = 6, p-값 = 0.595\n→ 거래일 독립성 가정과 일관됨',
             fontsize=12.5, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 850)

save('fig_6_4_sp500.png')


# ============================================================
# 그림 6.5: iPod 실험 결과 시각화
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 왼쪽: 막대 그래프
ax = axes[0]
groups = ['일반\n질문', '긍정\n가정', '부정\n가정']
disclosed = [2, 23, 36]
total = [73, 73, 73]
percent = [d/t*100 for d, t in zip(disclosed, total)]

bars = ax.bar(groups, percent, color=['#A0AEC0', '#D69E2E', '#C53030'],
              alpha=0.85, edgecolor='black', linewidth=1.5)
for bar, p, d in zip(bars, percent, disclosed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{p:.1f}%\n({d}/73)', ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('결함 공개율 (%)', fontsize=12)
ax.set_title('iPod 결함 공개율: 질문 형식별', fontsize=13, fontweight='bold')
ax.set_ylim(0, 60)
ax.grid(alpha=0.3, axis='y')

# 오른쪽: 카이제곱 분포와 관측 X²
ax = axes[1]
x = np.linspace(0, 50, 500)
y = stats.chi2.pdf(x, 2)
ax.plot(x, y, color='#2C5282', linewidth=2)
ax.fill_between(x, 0, y, alpha=0.1, color='#3182CE')

obs_x2 = 40.13
ax.axvline(obs_x2, color='#C53030', linewidth=2)
ax.fill_between(x[x >= obs_x2], 0, y[x >= obs_x2], alpha=0.6, color='#C53030')

ax.set_xlabel('X²')
ax.set_ylabel('밀도')
ax.set_title(f'카이제곱 (df=2): 관측 X² = {obs_x2}\np-값 ≈ 2×10⁻⁹ (매우 작음)',
             fontsize=12.5, fontweight='bold')
ax.set_xlim(0, 50)
ax.set_ylim(0, 0.5)
ax.annotate('극단적인 X²\n→ H₀ 강하게 기각',
            xy=(obs_x2, 0.005), xytext=(35, 0.2),
            fontsize=10.5, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFFFCC'),
            arrowprops=dict(arrowstyle='->'))
ax.grid(alpha=0.3)

plt.suptitle('iPod 결함 공개 실험: 질문 형식이 공개율에 매우 큰 영향',
             fontsize=14, fontweight='bold', y=1.02)
save('fig_6_5_ipod.png')


# ============================================================
# 그림 6.6: 결정나무 - 어떤 검정을 쓸 것인가
# ============================================================
fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

# 시작 노드
ax.add_patch(plt.Rectangle((5, 10.5), 4, 1, facecolor='#2C5282',
                            edgecolor='black', linewidth=1.5))
ax.text(7, 11, '범주형 자료\n추론 시작', ha='center', va='center',
        color='white', fontsize=11, fontweight='bold')

# 1단계 분기: 몇 개 그룹?
ax.annotate('', xy=(7, 9.5), xytext=(7, 10.4),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(7.3, 10, '그룹/범주 수?', fontsize=10, style='italic')

# 1 그룹
ax.add_patch(plt.Rectangle((0.5, 7.8), 3.5, 1.5, facecolor='#EBF8FF',
                            edgecolor='#2C5282', linewidth=1.5))
ax.text(2.25, 8.55, '1개 범주\n(이항)', ha='center', va='center', fontsize=10, fontweight='bold')
ax.annotate('', xy=(2.25, 7.9), xytext=(6, 9.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# 2 그룹
ax.add_patch(plt.Rectangle((5, 7.8), 4, 1.5, facecolor='#E6FFFA',
                            edgecolor='#319795', linewidth=1.5))
ax.text(7, 8.55, '2개 그룹\n(2 비율 비교)', ha='center', va='center', fontsize=10, fontweight='bold')
ax.annotate('', xy=(7, 7.9), xytext=(7, 10.4),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# 3+ 그룹
ax.add_patch(plt.Rectangle((10, 7.8), 3.5, 1.5, facecolor='#FFF5F5',
                            edgecolor='#742A2A', linewidth=1.5))
ax.text(11.75, 8.55, '3개 이상\n또는 분포 비교', ha='center', va='center', fontsize=10, fontweight='bold')
ax.annotate('', xy=(11.75, 7.9), xytext=(8, 9.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# 2단계: 각 시나리오의 검정 방법
# 1 그룹 → Z-검정
ax.add_patch(plt.Rectangle((0.2, 4.3), 4.1, 2.5, facecolor='#EBF8FF',
                            edgecolor='#2C5282', linewidth=1.2))
ax.text(2.25, 6.5, '6.1절', ha='center', fontsize=10, fontweight='bold', color='#2C5282')
ax.text(2.25, 5.95, '1-비율 Z-검정', ha='center', fontsize=10.5, fontweight='bold')
ax.text(2.25, 5.3,
        r'$Z=\dfrac{\hat{p}-p_0}{\sqrt{p_0(1-p_0)/n}}$',
        ha='center', fontsize=11)
ax.text(2.25, 4.6, '예: 백신 지지율 50%?', ha='center', fontsize=9, style='italic')
ax.annotate('', xy=(2.25, 6.7), xytext=(2.25, 7.8),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# 2 그룹 → Z 또는 카이제곱
ax.add_patch(plt.Rectangle((4.7, 4.3), 4.6, 2.5, facecolor='#E6FFFA',
                            edgecolor='#319795', linewidth=1.2))
ax.text(7, 6.5, '6.2절', ha='center', fontsize=10, fontweight='bold', color='#234E52')
ax.text(7, 5.95, '2-비율 Z-검정', ha='center', fontsize=10.5, fontweight='bold')
ax.text(7, 5.3,
        r'$Z=\dfrac{\hat{p}_1-\hat{p}_2}{SE_{\mathrm{pool}}}$',
        ha='center', fontsize=11)
ax.text(7, 4.6, '예: 남녀 흡연율 차이?', ha='center', fontsize=9, style='italic')
ax.annotate('', xy=(7, 6.7), xytext=(7, 7.8),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# 3+ → 카이제곱
ax.add_patch(plt.Rectangle((9.7, 4.3), 4.1, 2.5, facecolor='#FFF5F5',
                            edgecolor='#742A2A', linewidth=1.2))
ax.text(11.75, 6.5, '6.3 / 6.4절', ha='center', fontsize=10, fontweight='bold', color='#742A2A')
ax.text(11.75, 5.95, '카이제곱 검정', ha='center', fontsize=10.5, fontweight='bold')
ax.text(11.75, 5.3,
        r'$X^2=\sum\dfrac{(O-E)^2}{E}$',
        ha='center', fontsize=11)
ax.text(11.75, 4.6, '예: 주사위 공정, 독립성', ha='center', fontsize=9, style='italic')
ax.annotate('', xy=(11.75, 6.7), xytext=(11.75, 7.8),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# 3단계: 조건
ax.add_patch(plt.Rectangle((1, 1.5), 12, 2.3, facecolor='#FFFFE0',
                            edgecolor='#D69E2E', linewidth=1.5))
ax.text(7, 3.4, '공통 조건', ha='center', fontsize=11, fontweight='bold', color='#744210')
ax.text(7, 2.6, '① 독립성: 단순무작위표본 또는 무작위배정',
        ha='center', fontsize=10)
ax.text(7, 2.05, '② 표본크기 (Z-검정): 모든 그룹에서 np ≥ 10 그리고 n(1-p) ≥ 10',
        ha='center', fontsize=10)
ax.text(7, 1.7, '   (카이제곱): 모든 셀의 기대빈도 E ≥ 5',
        ha='center', fontsize=10)

ax.text(7, 0.6,
        '※ 조건이 위배되면: 시뮬레이션, 클로퍼-피어슨 구간, 피셔 정확검정 등 대안 사용',
        ha='center', fontsize=9.5, color='#666', style='italic')

plt.suptitle('제6장 결정나무: 범주형 자료에서 어떤 검정을 쓸 것인가?',
             fontsize=15, fontweight='bold', y=0.99)
save('fig_6_6_decision_tree.png')


# ============================================================
# 그림 6.7: 표본크기와 오차한계 관계
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))

ns = np.arange(50, 3001, 25)
me_50 = 1.96 * np.sqrt(0.5 * 0.5 / ns)
me_25 = 1.96 * np.sqrt(0.25 * 0.75 / ns)
me_10 = 1.96 * np.sqrt(0.1 * 0.9 / ns)

ax.plot(ns, me_50 * 100, label='p = 0.5 (최악 시나리오)', color='#C53030', linewidth=2.2)
ax.plot(ns, me_25 * 100, label='p = 0.25', color='#D69E2E', linewidth=2.2)
ax.plot(ns, me_10 * 100, label='p = 0.10', color='#38A169', linewidth=2.2)

ax.axhline(3, color='gray', linestyle='--', alpha=0.7)
ax.text(2700, 3.3, 'ME = 3%', fontsize=10, color='gray')

ax.set_xlabel('표본크기 n', fontsize=12)
ax.set_ylabel('95% 오차한계 (%)', fontsize=12)
ax.set_title('표본크기와 오차한계: 1/√n 비율로 줄어든다',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(50, 3000)
ax.set_ylim(0, 15)

save('fig_6_7_sample_size.png')


print('=' * 60)
print('Chapter 6 figures generated successfully')
print(f'Saved in: {OUTPUT_DIR}')
print('=' * 60)
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
    print(f'  {f} ({size:.1f} KB)')
