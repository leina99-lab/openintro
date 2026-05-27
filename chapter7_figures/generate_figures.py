"""
제7장 그림 생성 스크립트
수치형 자료에 대한 추론 - t-분포의 기초 개념을 시각화한다.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = '/home/claude/ch7/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), facecolor='white')
    plt.close()


# ============================================================
# 그림 7.0.1: 5/6장과 7장의 추론 도구 비교
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# 5/6장 - 비율
ax.add_patch(plt.Rectangle((0.5, 5), 6, 3.5, facecolor='#EBF8FF',
                            edgecolor='#2C5282', linewidth=2))
ax.text(3.5, 8, '5장 / 6장', ha='center', fontsize=14, fontweight='bold', color='#2C5282')
ax.text(3.5, 7.4, '범주형 자료 (비율)', ha='center', fontsize=11)
ax.text(3.5, 6.6, '• 점추정: p̂', ha='center', fontsize=10.5)
ax.text(3.5, 6.1, '• 표집분포: 정규분포', ha='center', fontsize=10.5)
ax.text(3.5, 5.6, '• SE = √(p(1-p)/n)  ※ p 알아야 함', ha='center', fontsize=10.5)
ax.text(3.5, 5.2, '• 검정통계량: Z', ha='center', fontsize=10.5, fontweight='bold')

# 7장 - 평균
ax.add_patch(plt.Rectangle((7.5, 5), 6, 3.5, facecolor='#FFF5F5',
                            edgecolor='#C53030', linewidth=2))
ax.text(10.5, 8, '7장', ha='center', fontsize=14, fontweight='bold', color='#742A2A')
ax.text(10.5, 7.4, '수치형 자료 (평균)', ha='center', fontsize=11)
ax.text(10.5, 6.6, '• 점추정: x̄', ha='center', fontsize=10.5)
ax.text(10.5, 6.1, '• 표집분포: t-분포', ha='center', fontsize=10.5)
ax.text(10.5, 5.6, '• SE = s/√n  ※ s로 σ 추정', ha='center', fontsize=10.5)
ax.text(10.5, 5.2, '• 검정통계량: T', ha='center', fontsize=10.5, fontweight='bold')

# 화살표
ax.annotate('', xy=(7.4, 7), xytext=(6.6, 7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax.text(7, 7.4, '확장', ha='center', fontsize=10, style='italic', color='#555')

# 공통 PCCC
ax.add_patch(plt.Rectangle((1, 1.5), 12, 3, facecolor='#FFFFE0',
                            edgecolor='#D69E2E', linewidth=1.5))
ax.text(7, 4, '공통: 가설검정 PCCC 4단계',
        ha='center', fontsize=13, fontweight='bold', color='#744210')
ax.text(7, 3.2, '준비(Prepare) → 확인(Check) → 계산(Calculate) → 결론(Conclude)',
        ha='center', fontsize=11)
ax.text(7, 2.4, '5장 §5.0에서 배운 p-값, α, CI 해석, 양측/단측, 1·2종 오류, 검정력은',
        ha='center', fontsize=10)
ax.text(7, 1.95, '7장의 모든 절에서 그대로 사용된다. 분포만 정규 → t로 바뀔 뿐.',
        ha='center', fontsize=10)

plt.suptitle('5/6장과 7장의 관계: 도구는 바뀌어도 사고의 흐름은 동일',
             fontsize=15, fontweight='bold', y=0.97)
save('fig_7_0_1_overview.png')


# ============================================================
# 그림 7.0.2: 7장의 4가지 시나리오
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

scenarios = [
    ("(a) 단일 표본 평균", "Q: 모평균 μ가 특정 값과 같은가?",
     "예: 평균 완주시간이\n2006년 93.29분과 같은가?",
     "단일 표본 t-검정\nT = (x̄ - μ₀)/(s/√n)\ndf = n - 1", "§7.1", '#3182CE'),
    ("(b) 대응표본 평균 차이", "Q: 같은 대상의 두 측정값에\n차이가 있는가?",
     "예: 같은 책의 두 사이트 가격\n→ 차이 = 가격₁ - 가격₂",
     "대응 t-검정\n(차이에 단일 표본 t)\ndf = n_diff - 1", "§7.2", '#38A169'),
    ("(c) 독립 두 표본 평균 차이", "Q: 두 모집단의 평균이\n다른가?",
     "예: 흡연모 vs 비흡연모의\n신생아 출생 체중",
     "독립 두 표본 t-검정\nT = (x̄₁-x̄₂)/SE\ndf ≈ min(n₁-1, n₂-1)", "§7.3", '#D69E2E'),
    ("(d) 다수 그룹 평균 비교", "Q: 3개 이상의 그룹 평균이\n모두 같은가?",
     "예: 야구 포지션별 출루율\n(외야, 내야, 포수)",
     "ANOVA (F-검정)\nF = MSG/MSE\ndf₁=k-1, df₂=n-k", "§7.5", '#C53030'),
]

for ax, (title, q, data, test, section, color) in zip(axes.flat, scenarios):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    rect = plt.Rectangle((0.3, 0.3), 9.4, 9.4, facecolor=color, alpha=0.08,
                          edgecolor=color, linewidth=2)
    ax.add_patch(rect)

    ax.text(5, 9, title, ha='center', va='center', fontsize=13, fontweight='bold', color=color)
    ax.text(5, 8, section, ha='center', va='center', fontsize=10, color=color, style='italic')
    ax.text(5, 6.7, q, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color))
    ax.text(5, 4.7, data, ha='center', va='center', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7FAFC', edgecolor='gray'))
    ax.text(5, 2, test, ha='center', va='center', fontsize=10, color=color, fontweight='bold')

plt.suptitle('제7장 개요: 수치형 자료에 대한 4가지 추론 시나리오',
             fontsize=15, fontweight='bold', y=1.00)
save('fig_7_0_2_scenarios.png')


# ============================================================
# 그림 7.0.3: 왜 t-분포인가 - σ vs s
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 왼쪽: σ 알 때 (이상적 세계)
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.add_patch(plt.Rectangle((0.5, 0.5), 9, 9, facecolor='#EBF8FF',
                            edgecolor='#3182CE', linewidth=2.5))
ax.text(5, 9, '이상적 세계 (σ 알 때)', ha='center', fontsize=13, fontweight='bold', color='#2C5282')
ax.text(5, 7.5, r'$SE = \dfrac{\sigma}{\sqrt{n}}$', ha='center', fontsize=15)
ax.text(5, 6, r'$Z = \dfrac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$', ha='center', fontsize=14)
ax.text(5, 4.3, '표집분포: 정확히 정규분포', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 3.2, 'σ는 알려진 상수이므로\n추가 불확실성이 없다.', ha='center', fontsize=10.5)
ax.text(5, 1.5, '※ 현실에서는 σ를 모름!', ha='center', fontsize=10.5, style='italic', color='#742A2A')

# 오른쪽: σ 모르고 s 쓸 때 (현실)
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.add_patch(plt.Rectangle((0.5, 0.5), 9, 9, facecolor='#FFF5F5',
                            edgecolor='#C53030', linewidth=2.5))
ax.text(5, 9, '현실 (σ 모름, s로 대체)', ha='center', fontsize=13, fontweight='bold', color='#742A2A')
ax.text(5, 7.5, r'$SE = \dfrac{s}{\sqrt{n}}$', ha='center', fontsize=15)
ax.text(5, 6, r'$T = \dfrac{\bar{x} - \mu_0}{s/\sqrt{n}}$', ha='center', fontsize=14)
ax.text(5, 4.3, '표집분포: t-분포 (df = n-1)', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 3.2, 's 자체도 표본마다 변동하므로\n추가 불확실성이 발생한다.', ha='center', fontsize=10.5)
ax.text(5, 1.5, '→ 정규분포보다 꼬리가 두꺼움', ha='center', fontsize=10.5, fontweight='bold', color='#742A2A')

plt.suptitle('왜 t-분포를 쓰는가?  σ를 모르고 s를 대신 쓰기 때문에',
             fontsize=14, fontweight='bold', y=1.02)
save('fig_7_0_3_why_t.png')


# ============================================================
# 그림 7.0.4: t-분포와 정규분포의 꼬리 비교
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))

x = np.linspace(-5, 5, 500)
ax.plot(x, stats.norm.pdf(x), label='정규분포 N(0,1)', color='black', linewidth=2.5, linestyle='--')
ax.plot(x, stats.t.pdf(x, 1), label='t-분포 (df=1)', color='#C53030', linewidth=2)
ax.plot(x, stats.t.pdf(x, 5), label='t-분포 (df=5)', color='#D69E2E', linewidth=2)
ax.plot(x, stats.t.pdf(x, 30), label='t-분포 (df=30)', color='#3182CE', linewidth=2)

ax.set_xlabel('값', fontsize=12)
ax.set_ylabel('밀도', fontsize=12)
ax.set_title('t-분포 vs 정규분포: 자유도(df)가 커질수록 정규분포에 가까워짐',
             fontsize=13.5, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.45)

# 꼬리 강조 주석
ax.annotate('df가 작을수록\n꼬리가 더 두껍다\n(불확실성 큼)',
            xy=(3, 0.04), xytext=(3.5, 0.2),
            fontsize=10.5, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFFFCC'),
            arrowprops=dict(arrowstyle='->', color='gray'))

save('fig_7_0_4_t_tails.png')


# ============================================================
# 그림 7.0.5: 자유도(df)의 직관적 이해
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# 그림 제목
ax.text(7, 5.5, '자유도(df = n-1)의 직관: 왜 n이 아니라 n-1인가?',
        ha='center', fontsize=14, fontweight='bold')

# 박스 1: 5개 수
ax.add_patch(plt.Rectangle((0.5, 2.5), 4.5, 2.3, facecolor='#EBF8FF',
                            edgecolor='#2C5282', linewidth=1.5))
ax.text(2.75, 4.5, '5개의 수가 있다:', ha='center', fontsize=11, fontweight='bold')
ax.text(2.75, 3.9, '{ a, b, c, d, e }', ha='center', fontsize=12, family='monospace')
ax.text(2.75, 3.2, '5개가 자유롭게 변할 수 있음', ha='center', fontsize=10)
ax.text(2.75, 2.75, '→ 자유도 5', ha='center', fontsize=10, fontweight='bold', color='#2C5282')

# 박스 2: 평균을 고정하면
ax.add_patch(plt.Rectangle((5.5, 2.5), 4.5, 2.3, facecolor='#FFF5F5',
                            edgecolor='#C53030', linewidth=1.5))
ax.text(7.75, 4.5, '평균을 x̄로 고정하면?', ha='center', fontsize=11, fontweight='bold')
ax.text(7.75, 3.9, 'a + b + c + d + e = 5x̄', ha='center', fontsize=11, family='monospace')
ax.text(7.75, 3.2, '4개는 자유, 1개는 종속', ha='center', fontsize=10)
ax.text(7.75, 2.75, '→ 자유도 4 (= n-1)', ha='center', fontsize=10, fontweight='bold', color='#742A2A')

# 박스 3: 결론
ax.add_patch(plt.Rectangle((10.5, 2.5), 3, 2.3, facecolor='#E6FFFA',
                            edgecolor='#319795', linewidth=1.5))
ax.text(12, 4.5, '핵심', ha='center', fontsize=11, fontweight='bold')
ax.text(12, 3.7, 's 계산에 x̄을\n이미 사용했으므로', ha='center', fontsize=10)
ax.text(12, 3, '"실질 정보 = n-1"', ha='center', fontsize=10.5, fontweight='bold', color='#234E52')

ax.text(7, 1.2,
        '※ 이것이 표본 분산 s² = Σ(xᵢ - x̄)² / (n-1) 에서 분모가 n이 아닌 n-1인 이유이기도 하다.',
        ha='center', fontsize=10.5, style='italic', color='#555')
ax.text(7, 0.5,
        '   "관측치 n개 - 추정에 쓴 모수 1개(x̄) = 남은 자유도 n-1"',
        ha='center', fontsize=10.5, style='italic', color='#555')

save('fig_7_0_5_df_intuition.png')


# ============================================================
# 그림 7.0.6: 표본크기별 정규성 점검 기준
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

ax.text(7, 6.5, 'x̄에 t-분포를 쓰기 위한 정규성 점검 (표본크기별 가이드)',
        ha='center', fontsize=14, fontweight='bold')

# n < 30
ax.add_patch(plt.Rectangle((0.5, 2.5), 4.2, 3.3, facecolor='#FFF5F5',
                            edgecolor='#C53030', linewidth=2))
ax.text(2.6, 5.4, 'n < 30', ha='center', fontsize=13, fontweight='bold', color='#742A2A')
ax.text(2.6, 4.7, '자료 자체가 거의\n정규분포여야 함', ha='center', fontsize=11)
ax.text(2.6, 3.7, '점검:\n• 히스토그램\n• Q-Q 플롯\n• 명확한 이상치 확인', ha='center', fontsize=10)

# 30 <= n < 60
ax.add_patch(plt.Rectangle((5, 2.5), 4, 3.3, facecolor='#FFFFE0',
                            edgecolor='#D69E2E', linewidth=2))
ax.text(7, 5.4, '30 ≤ n < 60', ha='center', fontsize=13, fontweight='bold', color='#744210')
ax.text(7, 4.5, '강한 비대칭이나\n극단적 이상치만 점검', ha='center', fontsize=11)
ax.text(7, 3.3, 'CLT가 어느 정도 작동하지만\n여전히 분포 형태 확인 필요', ha='center', fontsize=10)

# n >= 60
ax.add_patch(plt.Rectangle((9.3, 2.5), 4.2, 3.3, facecolor='#E6FFFA',
                            edgecolor='#319795', linewidth=2))
ax.text(11.4, 5.4, 'n ≥ 60', ha='center', fontsize=13, fontweight='bold', color='#234E52')
ax.text(11.4, 4.5, '극단적 이상치만\n확인하면 충분', ha='center', fontsize=11)
ax.text(11.4, 3.3, 'CLT가 강하게 작동하여\n원래 분포가 비정규여도 OK', ha='center', fontsize=10)

ax.text(7, 1.6, '※ 이상치는 단 1∼2개여도 표본 평균과 표준편차에 큰 영향을 미친다.',
        ha='center', fontsize=10.5, style='italic', color='#555')
ax.text(7, 0.9, '   이상치가 있다면 robust statistics(중앙값, IQR 등)를 고려한다.',
        ha='center', fontsize=10.5, style='italic', color='#555')

save('fig_7_0_6_normality_check.png')


# ============================================================
# 그림 7.0.7: 대응표본 vs 독립 두 표본 - 결정나무
# ============================================================
fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')

# 시작
ax.add_patch(plt.Rectangle((5, 9.5), 4, 1, facecolor='#2C5282',
                            edgecolor='black', linewidth=1.5))
ax.text(7, 10, '두 평균을 비교하려 한다', ha='center', va='center',
        color='white', fontsize=11.5, fontweight='bold')

# 1단계 질문
ax.annotate('', xy=(7, 9.2), xytext=(7, 9.4),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.add_patch(plt.Rectangle((4, 8), 6, 1, facecolor='#FFFFE0',
                            edgecolor='#D69E2E', linewidth=1.5))
ax.text(7, 8.5, '두 측정값이 자연스럽게 쌍을 이루는가?',
        ha='center', va='center', fontsize=11, fontweight='bold')

# 예 (대응)
ax.annotate('', xy=(3, 6.5), xytext=(5.5, 7.9),
            arrowprops=dict(arrowstyle='->', color='#38A169', lw=1.8))
ax.text(3.3, 7.2, '예', fontsize=12, color='#22543D', fontweight='bold')

ax.add_patch(plt.Rectangle((0.5, 5.3), 5, 1.5, facecolor='#E6FFFA',
                            edgecolor='#319795', linewidth=1.8))
ax.text(3, 6.4, '대응표본 (paired)', ha='center', fontsize=12, fontweight='bold', color='#234E52')
ax.text(3, 5.7, '§7.2', ha='center', fontsize=10, style='italic')

# 대응의 예시
ax.add_patch(plt.Rectangle((0.5, 2.8), 5, 2.2, facecolor='#F0FFF4',
                            edgecolor='#22543D', linewidth=1.2))
ax.text(3, 4.6, '예시:', ha='center', fontsize=10, fontweight='bold')
ax.text(3, 4, '• 같은 사람의 사전/사후 측정\n• 쌍둥이 비교\n• 같은 책의 두 사이트 가격\n• 좌/우 눈의 시력', ha='center', fontsize=9.5)

# 분석법
ax.add_patch(plt.Rectangle((0.5, 0.5), 5, 1.8, facecolor='#FFF5F5',
                            edgecolor='#C53030', linewidth=1.5))
ax.text(3, 1.8, '분석법', ha='center', fontsize=11, fontweight='bold', color='#742A2A')
ax.text(3, 1.05, '차이 dᵢ = x₁ᵢ - x₂ᵢ 계산 후\n그 차이에 단일 표본 t-검정 적용',
        ha='center', fontsize=10)

# 아니오 (독립)
ax.annotate('', xy=(11, 6.5), xytext=(8.5, 7.9),
            arrowprops=dict(arrowstyle='->', color='#C53030', lw=1.8))
ax.text(10.5, 7.2, '아니오', fontsize=12, color='#742A2A', fontweight='bold')

ax.add_patch(plt.Rectangle((8.5, 5.3), 5, 1.5, facecolor='#FFF5F5',
                            edgecolor='#C53030', linewidth=1.8))
ax.text(11, 6.4, '독립 두 표본 (independent)', ha='center', fontsize=12, fontweight='bold', color='#742A2A')
ax.text(11, 5.7, '§7.3', ha='center', fontsize=10, style='italic')

# 독립의 예시
ax.add_patch(plt.Rectangle((8.5, 2.8), 5, 2.2, facecolor='#FED7D7',
                            edgecolor='#822727', linewidth=1.2))
ax.text(11, 4.6, '예시:', ha='center', fontsize=10, fontweight='bold')
ax.text(11, 4,
        '• 처치군 vs 대조군 (다른 사람)\n• 남자 vs 여자\n• A학교 학생 vs B학교 학생\n• 도시 vs 농촌',
        ha='center', fontsize=9.5)

# 분석법
ax.add_patch(plt.Rectangle((8.5, 0.5), 5, 1.8, facecolor='#FFF5F5',
                            edgecolor='#C53030', linewidth=1.5))
ax.text(11, 1.8, '분석법', ha='center', fontsize=11, fontweight='bold', color='#742A2A')
ax.text(11, 1.05, '독립 두 표본 t-검정\nSE = √(s₁²/n₁ + s₂²/n₂)',
        ha='center', fontsize=10)

plt.suptitle('두 평균 비교의 결정나무: 대응표본인가, 독립표본인가?',
             fontsize=15, fontweight='bold', y=0.98)
save('fig_7_0_7_paired_vs_independent.png')


# ============================================================
# 그림 7.1: 단일 표본 t-신뢰구간 시각화 (돌고래 수은)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))

ci_lower, ci_upper = 3.29, 5.51
mean = 4.4

ax.plot([ci_lower, ci_upper], [1, 1], color='#2C5282', linewidth=4)
ax.scatter([ci_lower, ci_upper], [1, 1], color='#2C5282', s=100, zorder=5)
ax.scatter([mean], [1], color='#C53030', s=200, marker='D', zorder=5, label=f'점추정 x̄={mean}')

ax.text(ci_lower, 0.75, f'{ci_lower}', ha='center', fontsize=10)
ax.text(ci_upper, 0.75, f'{ci_upper}', ha='center', fontsize=10)
ax.text(mean, 1.25, f'{mean}', ha='center', fontsize=10)

# 안전 기준선 가정
safety_threshold = 1.0
ax.axvline(safety_threshold, color='green', linestyle='--', alpha=0.7)
ax.text(safety_threshold + 0.05, 1.6, '(가정) 안전 기준', fontsize=9, color='green')

ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.4, 1.9)
ax.set_yticks([])
ax.set_xlabel('수은 함량 (μg/wet g)', fontsize=12)
ax.set_title('Risso 돌고래 근육의 평균 수은 함량 95% 신뢰구간 (n=19)\n→ 안전 기준을 크게 초과',
             fontsize=12.5, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3, axis='x')

save('fig_7_1_dolphin_ci.png')


# ============================================================
# 그림 7.2: 단일 표본 t-검정 시각화 (Cherry Blossom)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))

df = 99
x = np.linspace(-4, 4, 500)
y = stats.t.pdf(x, df)

ax.plot(x, y, color='#2C5282', linewidth=2)
ax.fill_between(x, 0, y, alpha=0.1, color='#3182CE')

T_obs = 2.37
ax.fill_between(x[x >= T_obs], 0, y[x >= T_obs], alpha=0.6, color='#C53030')
ax.fill_between(x[x <= -T_obs], 0, y[x <= -T_obs], alpha=0.6, color='#C53030')

ax.axvline(T_obs, color='black', linewidth=1.5, linestyle='--')
ax.axvline(-T_obs, color='black', linewidth=1.5, linestyle='--')

ax.annotate(f'T = {T_obs}\n(x̄ = 97.32 vs μ₀ = 93.29)',
            xy=(T_obs, 0.02), xytext=(3, 0.18),
            fontsize=10.5, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFFFCC'),
            arrowprops=dict(arrowstyle='->'))

ax.text(0, 0.25, '양측 p-값 = 0.020 < 0.05\n→ H₀ 기각\n2017년 평균이 2006년과 다름',
        ha='center', fontsize=10.5, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

ax.set_xlabel('T (귀무분포: t(df=99))', fontsize=12)
ax.set_ylabel('밀도', fontsize=12)
ax.set_title('Cherry Blossom Race 단일 표본 t-검정 양측검정',
             fontsize=13, fontweight='bold')
ax.set_xlim(-4, 4)
ax.grid(alpha=0.3)

save('fig_7_2_cherry_blossom.png')


# ============================================================
# 그림 7.3: ANOVA 신호 대 잡음 개념
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

np.random.seed(42)

# 왼쪽: 그룹 차이 감지 어려운 경우 (작은 F)
ax = axes[0]
data_hard = [np.random.normal(50, 10, 30) for _ in range(3)]
positions = [1, 2, 3]
bp = ax.boxplot(data_hard, positions=positions, widths=0.6, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#3182CE', '#38A169', '#D69E2E']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_xticks(positions)
ax.set_xticklabels(['그룹 A', '그룹 B', '그룹 C'])
ax.set_ylabel('값')
ax.set_title('어려운 경우: 그룹 내 변동 큼\n→ F 작음, p-값 큼', fontsize=12, fontweight='bold')
ax.set_ylim(10, 90)
ax.grid(alpha=0.3, axis='y')
ax.text(2, 80, '신호(평균 차이) << 잡음(그룹 내 산포)',
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='#FFFFCC'))

# 오른쪽: 그룹 차이 감지 쉬운 경우 (큰 F)
ax = axes[1]
data_easy = [np.random.normal(40, 3, 30), np.random.normal(50, 3, 30), np.random.normal(60, 3, 30)]
bp = ax.boxplot(data_easy, positions=positions, widths=0.6, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#3182CE', '#38A169', '#D69E2E']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_xticks(positions)
ax.set_xticklabels(['그룹 A', '그룹 B', '그룹 C'])
ax.set_ylabel('값')
ax.set_title('쉬운 경우: 그룹 내 변동 작음\n→ F 큼, p-값 작음', fontsize=12, fontweight='bold')
ax.set_ylim(10, 90)
ax.grid(alpha=0.3, axis='y')
ax.text(2, 80, '신호(평균 차이) >> 잡음(그룹 내 산포)',
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='#FFFFCC'))

plt.suptitle('ANOVA의 직관: F = (그룹 간 변동) / (그룹 내 변동) = 신호 / 잡음',
             fontsize=14, fontweight='bold', y=1.02)
save('fig_7_3_anova_signal_noise.png')


# ============================================================
# 그림 7.4: F-분포의 형태
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))

x = np.linspace(0, 6, 500)
ax.plot(x, stats.f.pdf(x, 2, 30), label='F(df₁=2, df₂=30)', color='#3182CE', linewidth=2)
ax.plot(x, stats.f.pdf(x, 5, 30), label='F(df₁=5, df₂=30)', color='#38A169', linewidth=2)
ax.plot(x, stats.f.pdf(x, 10, 30), label='F(df₁=10, df₂=30)', color='#D69E2E', linewidth=2)

# 기각역 (df₁=2 기준)
crit = stats.f.ppf(0.95, 2, 30)
ax.axvline(crit, color='#C53030', linestyle='--', linewidth=1.5)
ax.text(crit + 0.05, 0.7, f'α=0.05 임계값\nF*={crit:.2f}', fontsize=10, color='#742A2A')

ax.set_xlabel('F', fontsize=12)
ax.set_ylabel('밀도', fontsize=12)
ax.set_title('F-분포: 카이제곱처럼 항상 0 이상, 오른쪽 꼬리만 사용',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

ax.text(4, 0.4, 'ANOVA는 항상\n오른쪽 단측검정\n(카이제곱과 같은 이유)',
        ha='center', fontsize=10.5,
        bbox=dict(boxstyle='round', facecolor='#FFFFE0', edgecolor='#D69E2E'))

save('fig_7_4_f_distribution.png')


print('=' * 60)
print('Chapter 7 figures generated successfully')
print(f'Saved in: {OUTPUT_DIR}')
print('=' * 60)
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
    print(f'  {f} ({size:.1f} KB)')
