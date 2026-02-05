"""
Chapter 7: Inference for Numerical Data - 모든 그림 재생성
OpenIntro Statistics 원본 PDF의 모든 그림을 Python으로 재생성
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# 출력 디렉토리
OUTPUT_DIR = "/home/claude/ch7_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 스타일 설정
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {name}.png")

# =============================================================================
# Figure 7.1: t-distribution vs Normal distribution
# =============================================================================
def fig_7_1():
    """t-분포와 정규분포 비교"""
    x = np.linspace(-4.5, 4.5, 500)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, stats.norm.pdf(x), 'b--', lw=2, label='Normal')
    ax.plot(x, stats.t.pdf(x, 5), 'r-', lw=2, label='t-distribution')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title('Figure 7.1: Comparison of a t-distribution and a normal distribution')
    ax.grid(True, alpha=0.3)
    save_fig('fig_7_1_t_vs_normal')

# =============================================================================
# Figure 7.2: t-distributions with various degrees of freedom
# =============================================================================
def fig_7_2():
    """다양한 자유도의 t-분포"""
    x = np.linspace(-4.5, 4.5, 500)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, stats.norm.pdf(x), 'k-', lw=2, label='normal')
    ax.plot(x, stats.t.pdf(x, 8), 'b--', lw=1.5, label='t, df = 8')
    ax.plot(x, stats.t.pdf(x, 4), 'g-.', lw=1.5, label='t, df = 4')
    ax.plot(x, stats.t.pdf(x, 2), 'r:', lw=2, label='t, df = 2')
    ax.plot(x, stats.t.pdf(x, 1), 'm--', lw=1.5, label='t, df = 1')
    
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(0, 0.42)
    ax.legend(loc='upper right')
    ax.set_title('Figure 7.2: t-distributions with different degrees of freedom')
    ax.grid(True, alpha=0.3)
    save_fig('fig_7_2_t_distributions')

# =============================================================================
# Figure 7.3: t-distribution with df=18, shaded area below -2.10
# =============================================================================
def fig_7_3():
    """df=18인 t-분포, -2.10 아래 음영"""
    x = np.linspace(-4, 4, 500)
    df = 18
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, stats.t.pdf(x, df), 'b-', lw=2)
    
    x_fill = x[x <= -2.10]
    ax.fill_between(x_fill, stats.t.pdf(x_fill, df), alpha=0.4, color='steelblue')
    ax.axvline(x=-2.10, color='red', ls='--', lw=1)
    ax.text(-2.10, -0.02, '-2.10', ha='center', fontsize=10)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.4)
    ax.set_title('Figure 7.3: t-distribution (df=18), area below -2.10 = 0.025')
    ax.grid(True, alpha=0.3)
    save_fig('fig_7_3_t_df18_tail')

# =============================================================================
# Figure 7.4: Two t-distribution examples
# =============================================================================
def fig_7_4():
    """두 개의 t-분포 예제"""
    x = np.linspace(-4, 4, 500)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: df=20, area above 1.65
    df1 = 20
    axes[0].plot(x, stats.t.pdf(x, df1), 'b-', lw=2)
    x_fill = x[x >= 1.65]
    axes[0].fill_between(x_fill, stats.t.pdf(x_fill, df1), alpha=0.4, color='steelblue')
    axes[0].axvline(x=1.65, color='red', ls='--', lw=1)
    axes[0].set_title('df=20, area above 1.65')
    axes[0].set_xlim(-4, 4)
    axes[0].grid(True, alpha=0.3)
    
    # Right: df=2, area beyond ±3
    df2 = 2
    axes[1].plot(x, stats.t.pdf(x, df2), 'b-', lw=2)
    x_left = x[x <= -3]
    x_right = x[x >= 3]
    axes[1].fill_between(x_left, stats.t.pdf(x_left, df2), alpha=0.4, color='steelblue')
    axes[1].fill_between(x_right, stats.t.pdf(x_right, df2), alpha=0.4, color='steelblue')
    axes[1].axvline(x=-3, color='red', ls='--', lw=1)
    axes[1].axvline(x=3, color='red', ls='--', lw=1)
    axes[1].set_title('df=2, area beyond ±3')
    axes[1].set_xlim(-4, 4)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Figure 7.4: t-distribution tail area examples')
    save_fig('fig_7_4_t_examples')

# =============================================================================
# Figure 7.7: Cherry Blossom Race histogram
# =============================================================================
def fig_7_7():
    """Cherry Blossom Race 완주 시간 히스토그램"""
    np.random.seed(42)
    # 시뮬레이션: 평균 97.32, 표준편차 16.98
    times = np.random.normal(97.32, 16.98, 100)
    times = np.clip(times, 60, 150)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(times, bins=12, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=97.32, color='red', ls='-', lw=2, label=f'Mean = 97.32')
    ax.axvline(x=93.29, color='green', ls='--', lw=2, label=f'2006 avg = 93.29')
    
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Frequency')
    ax.set_title('Figure 7.7: Cherry Blossom Race 2017 sample (n=100)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig('fig_7_7_cherry_blossom')

# =============================================================================
# Figure 7.8: Textbook prices table visualization
# =============================================================================
def fig_7_8():
    """교과서 가격 비교 테이블"""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    data = [
        ['', 'dept', 'ucla_new', 'amaz_new', 'diff'],
        ['1', 'Am Ind', '47.97', '47.45', '0.52'],
        ['2', 'Anthro', '14.26', '13.55', '0.71'],
        ['3', 'Anthro', '13.50', '12.53', '0.97'],
        ['...', '...', '...', '...', '...'],
        ['68', 'tic Frn', '14.00', '14.00', '0.00']
    ]
    
    table = ax.table(cellText=data, loc='center', cellLoc='center',
                     colWidths=[0.1, 0.2, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax.set_title('Figure 7.8: Textbook prices (UCLA vs Amazon)', pad=20)
    save_fig('fig_7_8_textbook_table')

# =============================================================================
# Figure 7.9: Textbook price difference histogram
# =============================================================================
def fig_7_9():
    """교과서 가격 차이 히스토그램"""
    np.random.seed(123)
    # 시뮬레이션: 평균 3.58, 표준편차 13.42
    diff = np.random.normal(3.58, 13.42, 68)
    
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(diff, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='red', ls='--', lw=2, label='No difference')
    ax.axvline(x=3.58, color='green', ls='-', lw=2, label='Mean = $3.58')
    
    ax.set_xlabel('UCLA Bookstore Price − Amazon Price (USD)')
    ax.set_ylabel('Frequency')
    ax.set_title('Figure 7.9: Histogram of textbook price differences')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig('fig_7_9_textbook_diff')

# =============================================================================
# Figure 7.11 & 7.12: ESC study summary and histograms
# =============================================================================
def fig_7_11_12():
    """배아줄기세포 연구 히스토그램"""
    np.random.seed(42)
    esc = np.random.normal(3.50, 5.17, 9)
    control = np.random.normal(-4.33, 2.76, 9)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(esc, bins=5, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0, color='red', ls='--', lw=1.5)
    axes[0].axvline(x=3.50, color='green', ls='-', lw=2)
    axes[0].set_xlabel('Heart Pumping Change (%)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('ESC Treatment (n=9, mean=3.50%)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(control, bins=5, edgecolor='black', alpha=0.7, color='coral')
    axes[1].axvline(x=0, color='red', ls='--', lw=1.5)
    axes[1].axvline(x=-4.33, color='green', ls='-', lw=2)
    axes[1].set_xlabel('Heart Pumping Change (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Control (n=9, mean=-4.33%)')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Figure 7.12: ESC Study - Heart function change histograms')
    save_fig('fig_7_12_esc_study')

# =============================================================================
# Figure 7.14: Birth weight histograms
# =============================================================================
def fig_7_14():
    """출생 체중 히스토그램"""
    np.random.seed(42)
    smoker = np.random.normal(6.78, 1.43, 50)
    nonsmoker = np.random.normal(7.18, 1.60, 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(smoker, bins=10, edgecolor='black', alpha=0.7, color='salmon')
    axes[0].set_xlabel('Birth weight (pounds)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Smoker (n=50, mean={np.mean(smoker):.2f})')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(nonsmoker, bins=10, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1].set_xlabel('Birth weight (pounds)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Non-smoker (n=100, mean={np.mean(nonsmoker):.2f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Figure 7.14: NC Birth weights by smoking status')
    save_fig('fig_7_14_birthweight')

# =============================================================================
# Figure 7.15: Two-sample t-test sampling distribution
# =============================================================================
def fig_7_15():
    """두 표본 t-검정 표집분포"""
    x = np.linspace(-4, 4, 500)
    
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, stats.t.pdf(x, 48), 'b-', lw=2)
    
    # p-값 음영 (T=1.26)
    T_obs = 1.26
    x_right = x[x >= T_obs]
    x_left = x[x <= -T_obs]
    ax.fill_between(x_right, stats.t.pdf(x_right, 48), alpha=0.4, color='steelblue')
    ax.fill_between(x_left, stats.t.pdf(x_left, 48), alpha=0.4, color='steelblue')
    
    ax.axvline(x=0, color='green', ls='-', lw=1.5)
    ax.axvline(x=T_obs, color='red', ls='--', lw=1.5)
    ax.axvline(x=-T_obs, color='red', ls='--', lw=1.5)
    
    ax.text(0, -0.02, 'μ₀ = 0', ha='center')
    ax.text(T_obs, -0.02, f'T = {T_obs}', ha='center')
    
    ax.set_title('Figure 7.15: Sampling distribution for two-sample t-test')
    ax.set_xlim(-4, 4)
    ax.grid(True, alpha=0.3)
    save_fig('fig_7_15_two_sample_t')

# =============================================================================
# Figure 7.17: Exam version comparison
# =============================================================================
def fig_7_17():
    """시험 버전 비교 t-검정"""
    x = np.linspace(-4, 4, 500)
    df = 26
    T_obs = 1.15
    
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, stats.t.pdf(x, df), 'b-', lw=2)
    
    x_right = x[x >= T_obs]
    x_left = x[x <= -T_obs]
    ax.fill_between(x_right, stats.t.pdf(x_right, df), alpha=0.4, color='steelblue')
    ax.fill_between(x_left, stats.t.pdf(x_left, df), alpha=0.4, color='steelblue')
    
    ax.axvline(x=T_obs, color='red', ls='--', lw=1.5)
    ax.text(T_obs + 0.2, 0.15, f'T = {T_obs}', fontsize=11)
    
    p_val = 2 * (1 - stats.t.cdf(T_obs, df))
    ax.set_title(f'Figure 7.17: Exam version t-test (df={df}, p-value={p_val:.2f})')
    ax.set_xlim(-4, 4)
    ax.grid(True, alpha=0.3)
    save_fig('fig_7_17_exam_version')

# =============================================================================
# Figure 7.18: Power calculation visualization
# =============================================================================
def fig_7_18():
    """검정력 계산 시각화"""
    SE = 2.0
    null_mean = 0
    alt_mean = -3
    crit = 1.96 * SE
    
    x = np.linspace(-10, 8, 500)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 귀무분포
    null_dist = stats.norm.pdf(x, null_mean, SE)
    ax.plot(x, null_dist, 'b-', lw=2, label='Null distribution (μ=0)')
    
    # 대립분포
    alt_dist = stats.norm.pdf(x, alt_mean, SE)
    ax.plot(x, alt_dist, 'r-', lw=2, label=f'Alternative distribution (μ={alt_mean})')
    
    # 기각역
    ax.axvline(x=-crit, color='green', ls='--', lw=1.5)
    ax.axvline(x=crit, color='green', ls='--', lw=1.5)
    
    # 검정력 영역 (대립분포에서 기각역)
    x_power = x[x <= -crit]
    ax.fill_between(x_power, stats.norm.pdf(x_power, alt_mean, SE), 
                    alpha=0.3, color='red', label='Power')
    
    ax.legend()
    ax.set_title('Figure 7.18: Power calculation for two-sample test')
    ax.set_xlabel('x̄₁ - x̄₂')
    ax.grid(True, alpha=0.3)
    save_fig('fig_7_18_power_calc')

# =============================================================================
# Figure 7.19: ANOVA concept - side-by-side dot plots
# =============================================================================
def fig_7_19():
    """ANOVA 개념 설명 - 점도표"""
    np.random.seed(42)
    
    # 그룹 I, II, III: 큰 그룹 내 변동
    g1 = np.random.normal(1.5, 1.2, 30)
    g2 = np.random.normal(1.7, 1.2, 30)
    g3 = np.random.normal(1.3, 1.2, 30)
    
    # 그룹 IV, V, VI: 작은 그룹 내 변동
    g4 = np.random.normal(1.0, 0.25, 30)
    g5 = np.random.normal(2.0, 0.25, 30)
    g6 = np.random.normal(1.5, 0.25, 30)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    groups = [g1, g2, g3, g4, g5, g6]
    labels = ['I', 'II', 'III', 'IV', 'V', 'VI']
    colors = ['C0', 'C0', 'C0', 'C1', 'C1', 'C1']
    
    for i, (data, label, color) in enumerate(zip(groups, labels, colors)):
        x_pos = np.random.uniform(i+0.7, i+1.3, len(data))
        ax.scatter(x_pos, data, alpha=0.6, s=30, c=color)
        ax.hlines(np.mean(data), i+0.6, i+1.4, colors='red', lw=2)
    
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Outcome')
    ax.set_title('Figure 7.19: ANOVA concept - comparing group variability')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 주석
    ax.annotate('High within-group\nvariability', xy=(2, 4), fontsize=10, ha='center')
    ax.annotate('Low within-group\nvariability', xy=(5, 2.8), fontsize=10, ha='center')
    
    save_fig('fig_7_19_anova_concept')

# =============================================================================
# Figure 7.23: MLB OBP by position - box plot
# =============================================================================
def fig_7_23():
    """MLB 포지션별 출루율 상자그림"""
    np.random.seed(42)
    
    of_data = np.random.normal(0.319, 0.038, 160)
    if_data = np.random.normal(0.320, 0.039, 226)
    c_data = np.random.normal(0.302, 0.038, 43)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    bp = ax.boxplot([of_data, if_data, c_data], 
                    tick_labels=['OF (Outfield)', 'IF (Infield)', 'C (Catcher)'],
                    patch_artist=True)
    
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('On-Base Percentage (OBP)')
    ax.set_xlabel('Position')
    ax.set_title('Figure 7.23: MLB OBP by Position (2018)')
    ax.grid(True, alpha=0.3, axis='y')
    save_fig('fig_7_23_mlb_boxplot')

# =============================================================================
# Figure 7.24: F-distribution
# =============================================================================
def fig_7_24():
    """F-분포와 p-값"""
    df1, df2 = 2, 426
    F_obs = 5.077
    
    x = np.linspace(0, 10, 500)
    
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, stats.f.pdf(x, df1, df2), 'b-', lw=2)
    
    x_fill = x[x >= F_obs]
    ax.fill_between(x_fill, stats.f.pdf(x_fill, df1, df2), alpha=0.4, color='red')
    
    ax.axvline(x=F_obs, color='red', ls='--', lw=1.5)
    
    p_val = 1 - stats.f.cdf(F_obs, df1, df2)
    ax.text(F_obs + 0.3, 0.4, f'F = {F_obs}\np = {p_val:.4f}', fontsize=11)
    ax.text(3, 0.6, 'Small tail area', fontsize=10)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('F')
    ax.set_title(f'Figure 7.24: F-distribution (df₁={df1}, df₂={df2})')
    ax.grid(True, alpha=0.3)
    save_fig('fig_7_24_f_distribution')

# =============================================================================
# Figure 7.25: ANOVA residuals and normal Q-Q plot
# =============================================================================
def fig_7_25():
    """ANOVA 잔차 진단 플롯"""
    np.random.seed(42)
    
    # 시뮬레이션 잔차
    residuals = np.random.normal(0, 0.04, 429)
    fitted = np.random.uniform(0.28, 0.36, 429)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 잔차 vs 적합값
    axes[0].scatter(fitted, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='red', ls='--', lw=1.5)
    axes[0].set_xlabel('Fitted values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Normal Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Figure 7.25: ANOVA diagnostic plots')
    save_fig('fig_7_25_anova_diagnostics')

# =============================================================================
# Figure 7.27 & 7.28: Midterm scores ANOVA
# =============================================================================
def fig_7_27_28():
    """중간고사 점수 ANOVA"""
    np.random.seed(42)
    
    classA = np.random.normal(75.1, 13.9, 58)
    classB = np.random.normal(72.0, 13.8, 55)
    classC = np.random.normal(78.9, 13.1, 51)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    bp = ax.boxplot([classA, classB, classC], 
                    tick_labels=['A', 'B', 'C'],
                    patch_artist=True)
    
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Midterm Scores')
    ax.set_xlabel('Lecture')
    ax.set_title('Figure 7.28: Midterm scores by lecture section')
    ax.grid(True, alpha=0.3, axis='y')
    save_fig('fig_7_28_midterm_boxplot')

# =============================================================================
# Example 7.1 histograms
# =============================================================================
def fig_example_7_1():
    """예제 7.1: 정규성 조건 점검"""
    np.random.seed(42)
    
    sample1 = np.random.exponential(2, 15)
    sample2 = np.concatenate([np.random.normal(8, 3, 49), [25]])  # 이상치 포함
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(sample1, bins=6, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Sample 1 Observations (n=15)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Sample 1: No clear outliers ✓')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(sample2, bins=10, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Sample 2 Observations (n=50)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Sample 2: Extreme outlier present ✗')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Example 7.1: Checking normality condition')
    save_fig('fig_example_7_1_normality')

# =============================================================================
# Power curve
# =============================================================================
def fig_power_curve():
    """검정력 곡선"""
    def calc_power(n, effect=3, sigma=12, alpha=0.05):
        SE = np.sqrt(2 * sigma**2 / n)
        z_crit = stats.norm.ppf(1 - alpha/2)
        boundary = z_crit * SE
        power = stats.norm.cdf(-boundary, loc=effect, scale=SE)
        power += 1 - stats.norm.cdf(boundary, loc=effect, scale=SE)
        return power
    
    n_values = np.arange(20, 1001, 10)
    powers = [calc_power(n) for n in n_values]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_values, powers, 'b-', lw=2)
    ax.axhline(y=0.80, color='red', ls='--', lw=1.5, label='80% Power')
    ax.axhline(y=0.90, color='orange', ls='--', lw=1.5, label='90% Power')
    
    # 80% 지점
    n_80 = 251
    ax.plot(n_80, 0.80, 'ro', markersize=8)
    ax.annotate(f'n ≈ {n_80}', xy=(n_80, 0.80), xytext=(n_80+80, 0.72),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlabel('Sample size per group (n)')
    ax.set_ylabel('Power')
    ax.set_title('Power curve (effect=3, σ=12, α=0.05)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    save_fig('fig_power_curve')

# =============================================================================
# Hypothesis test diagram
# =============================================================================
def fig_hypothesis_test():
    """가설검정 도식"""
    x = np.linspace(-4, 4, 500)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, stats.norm.pdf(x), 'b-', lw=2)
    
    # 기각역
    ax.fill_between(x[x <= -1.96], stats.norm.pdf(x[x <= -1.96]), 
                    alpha=0.4, color='red', label='Rejection region (α/2)')
    ax.fill_between(x[x >= 1.96], stats.norm.pdf(x[x >= 1.96]), 
                    alpha=0.4, color='red')
    
    ax.axvline(x=-1.96, color='red', ls='--', lw=1.5)
    ax.axvline(x=1.96, color='red', ls='--', lw=1.5)
    ax.axvline(x=0, color='green', ls='-', lw=1.5)
    
    ax.text(-1.96, -0.03, '-1.96', ha='center')
    ax.text(1.96, -0.03, '1.96', ha='center')
    ax.text(0, -0.03, '0 (H₀)', ha='center')
    
    ax.set_title('Two-sided hypothesis test (α = 0.05)')
    ax.legend()
    ax.set_xlim(-4, 4)
    ax.grid(True, alpha=0.3)
    save_fig('fig_hypothesis_test')

# =============================================================================
# 모든 그림 생성
# =============================================================================
if __name__ == "__main__":
    print("Generating all Chapter 7 figures...")
    
    fig_7_1()
    fig_7_2()
    fig_7_3()
    fig_7_4()
    fig_7_7()
    fig_7_9()
    fig_7_11_12()
    fig_7_14()
    fig_7_15()
    fig_7_17()
    fig_7_18()
    fig_7_19()
    fig_7_23()
    fig_7_24()
    fig_7_25()
    fig_7_27_28()
    fig_example_7_1()
    fig_power_curve()
    fig_hypothesis_test()
    
    print(f"\nAll figures saved to {OUTPUT_DIR}")
    print(f"Total: {len(os.listdir(OUTPUT_DIR))} files")
