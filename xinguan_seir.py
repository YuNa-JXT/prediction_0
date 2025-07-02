import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# 设置中文字体，以确保图表中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
class SEIRModel:
    def __init__(self, N, I0, E0=0, R0=0, beta=0.5, sigma=0.2, gamma=0.1):
        self.N = N
        self.I0 = I0
        self.E0 = E0
        self.R0 = R0
        self.S0 = N - I0 - E0 - R0
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

    def deriv(self, y, t, beta, sigma, gamma, N):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def solve(self, t, intervention_day=None, reduction=0):
        y0 = [self.S0, self.E0, self.I0, self.R0]

        if intervention_day is None:
            sol = odeint(self.deriv, y0, t, args=(self.beta, self.sigma, self.gamma, self.N))
        else:
            t1 = t[t <= intervention_day]
            t2 = t[t > intervention_day]

            if len(t1) > 0:
                sol1 = odeint(self.deriv, y0, t1, args=(self.beta, self.sigma, self.gamma, self.N))
                if len(t2) > 0:
                    beta_new = self.beta * (1 - reduction)
                    y0_new = sol1[-1]
                    sol2 = odeint(self.deriv, y0_new, t2, args=(beta_new, self.sigma, self.gamma, self.N))
                    sol = np.vstack([sol1, sol2])
                else:
                    sol = sol1
            else:
                beta_new = self.beta * (1 - reduction)
                sol = odeint(self.deriv, y0, t, args=(beta_new, self.sigma, self.gamma, self.N))

        return sol

    def get_R0(self):
        return self.beta / self.gamma

    def simulate(self, days=365, intervention_day=None, reduction=0):
        t = np.linspace(0, days, days + 1)
        sol = self.solve(t, intervention_day, reduction)

        results = {
            'time': t,
            'S': sol[:, 0],
            'E': sol[:, 1],
            'I': sol[:, 2],
            'R': sol[:, 3]
        }

        stats = self.calculate_stats(results)
        return results, stats

    def calculate_stats(self, results):
        I = results['I']
        t = results['time']

        peak_infected = np.max(I)
        peak_day = t[np.argmax(I)]

        final_R = results['R'][-1]
        attack_rate = final_R / self.N * 100

        R0 = self.get_R0()

        return {
            'peak_infected': peak_infected,
            'peak_day': peak_day,
            'attack_rate': attack_rate,
            'R0': R0,
            'final_S': results['S'][-1],
            'final_E': results['E'][-1],
            'final_I': results['I'][-1],
            'final_R': results['R'][-1]
        }

def plot_seir(results, stats, title="SEIR模型疫情传播模拟"):
    fig = Figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(results['time'], results['S'], 'b-', label='易感者 (S)', linewidth=2)
    ax1.plot(results['time'], results['E'], 'orange', label='潜伏者 (E)', linewidth=2)
    ax1.plot(results['time'], results['I'], 'r-', label='感染者 (I)', linewidth=2)
    ax1.plot(results['time'], results['R'], 'g-', label='康复者 (R)', linewidth=2)
    ax1.set_xlabel('天数')
    ax1.set_ylabel('人数')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(results['time'], results['I'], 'r-', linewidth=2)
    ax2.axhline(y=stats['peak_infected'], color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=stats['peak_day'], color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('天数')
    ax2.set_ylabel('感染者人数')
    ax2.set_title(f'感染者变化 (峰值: {stats["peak_infected"]:.0f}人, 第{stats["peak_day"]:.0f}天)')
    ax2.grid(True, alpha=0.3)

    return fig

def main():
    st.title("SEIR模型疫情传播模拟")

    # 设置模型参数
    st.sidebar.header("模型参数")
    N = st.sidebar.number_input("总人口数", min_value=1000, value=1000000, step=1000)
    I0 = st.sidebar.number_input("初始感染者", min_value=1, value=100, step=1)
    E0 = st.sidebar.number_input("初始潜伏者", min_value=0, value=0, step=1)
    R0 = st.sidebar.number_input("初始康复者", min_value=0, value=0, step=1)
    beta = st.sidebar.slider("传播率 β", min_value=0.1, max_value=2.0, value=0.5, step=0.1, format="%.1f")
    sigma = st.sidebar.slider("潜伏期转感染率 σ (1/天)", min_value=0.1, max_value=1.0, value=0.2, step=0.05, format="%.2f")
    gamma = st.sidebar.slider("康复率 γ (1/天)", min_value=0.05, max_value=0.5, value=0.1, step=0.01, format="%.2f")
    intervention_day = st.sidebar.number_input("干预开始天数", min_value=0, value=60, step=1)
    reduction = st.sidebar.slider("传播率减少比例", min_value=0, max_value=90, value=50, step=5, format="%d%%")

    # 创建模型实例
    model = SEIRModel(N, I0, E0, R0, beta, sigma, gamma)

    # 运行模拟
    results, stats = model.simulate(intervention_day=intervention_day, reduction=reduction)

    # 绘制结果
    fig = plot_seir(results, stats)
    st.pyplot(fig)

    # 显示统计数据
    st.subheader("模拟结果")
    st.write(f"峰值感染人数: {stats['peak_infected']:.0f}")
    st.write(f"峰值出现时间: 第{stats['peak_day']:.0f}天")
    st.write(f"累计感染率: {stats['attack_rate']:.1f}%")
    st.write(f"基本再生数 R₀: {stats['R0']:.2f}")

    # 参数解释和重要性
    st.subheader("参数解释和重要性")
    st.markdown("**总人口数 (N)** ⭐⭐⭐⭐⭐")
    st.write("总人口规模会直接影响疫情传播的规模和趋势。人口越多,疫情的影响就越大。")
    st.markdown("**初始感染者 (I0) ** ⭐⭐⭐⭐")
    st.write("初始感染者数量决定了疫情的起始点。这个数值越大,疫情就越容易蔓延。")
    st.markdown("**传播率 (β) ** ⭐⭐⭐⭐⭐")
    st.write("传播率是最重要的参数,它决定了病毒的传播速度。这个值越高,疫情扩散就越快。")
    st.markdown("**潜伏期转感染率 (σ) ** ⭐⭐⭐")
    st.write("这个参数反映了潜伏期的长短,影响了疫情的早期发展。值越大,潜伏期越短。")
    st.markdown("**康复率 (γ) ** ⭐⭐⭐")
    st.write("这个参数决定了感染者的治愈速度。值越大,感染者越快康复。")
    st.markdown("**干预开始天数 ** ⭐⭐⭐⭐")
    st.write("这个参数决定了防控措施的启动时机,对疫情的发展轨迹有重要影响。")
    st.markdown("**传播率减少比例 ** ⭐⭐⭐⭐⭐")
    st.write("这个参数反映了防控措施的效果,决定了疫情受到多大程度的遏制。")

if __name__ == "__main__":
    main()