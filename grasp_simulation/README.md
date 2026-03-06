# 机器人抓取仿真环境 - Grasp Simulation

> 基于 PyTorch + Gym 的简化抓取仿真环境，用于 DQN 算法训练和验证  
> 作者：志哥的研究团队 | 日期：2026-03-06

---

## 📋 项目简介

本项目提供一个简化的 2D 平面抓取仿真环境，用于：
- **算法验证**：快速测试强化学习算法（如 DQN、PPO 等）
- **教学演示**：理解 VLA-Grasp 论文中的强化学习基础
- **基线对比**：作为 VLA 方法的简单基线

**注意**：这是简化版本，真实实验请使用 Isaac Sim 或 MuJoCo。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install torch gymnasium numpy
```

### 2. 运行训练

```bash
# 训练 DQN 智能体（500 回合）
python grasp_env.py

# 训练并渲染（每 50 回合显示一次）
python -c "from grasp_env import train_dqn; train_dqn(num_episodes=500, render=True)"
```

### 3. 测试结果

训练完成后会自动保存模型 `dqn_grasp_model.pth`，并运行测试：

```
========================================
测试训练好的模型
========================================
Step: 1
Object: pos=[0.23, -0.15], class=2
Arm: pos=[0.10, -0.20], gripper=open
Grasped: False
----------------------------------------
...
测试完成 | 总奖励：10.50 | 成功：True
```

---

## 📦 项目结构

```
grasp_simulation/
├── grasp_env.py          # 主代码文件
│   ├── GraspEnv          # 抓取环境类
│   ├── DQN               # DQN 网络
│   ├── ReplayBuffer      # 经验回放
│   ├── DQNAgent          # DQN 智能体
│   └── train_dqn         # 训练函数
├── README.md             # 本文件
├── requirements.txt      # 依赖列表
└── dqn_grasp_model.pth   # 训练好的模型（训练后生成）
```

---

## 🎮 环境说明

### 状态空间（10 维）

| 维度 | 含义 | 范围 |
|------|------|------|
| 0-1 | 物体位置 (x, y) | [-0.5, 0.5] 米 |
| 2-6 | 物体类别（one-hot） | 5 类物体 |
| 7-8 | 机械臂位置 (x, y) | [-0.5, 0.5] 米 |
| 9 | 夹爪状态 | 0=闭合，1=打开 |

### 动作空间（7 个离散动作）

| 动作 ID | 含义 |
|--------|------|
| 0 | 停止 |
| 1 | 左移 (-x) |
| 2 | 右移 (+x) |
| 3 | 上移 (+y) |
| 4 | 下移 (-y) |
| 5 | 闭合夹爪 |
| 6 | 打开夹爪 |

### 奖励函数

| 情况 | 奖励 |
|------|------|
| 成功抓取 | +10.0 |
| 接近物体（<20cm） | +0.1 × (1 - 距离/0.2) |
| 无效动作（停止） | -0.01 |
| 碰撞边界 | -1.0 |

### 终止条件

- **成功**：夹爪闭合时距离物体<10cm
- **失败**：超过 100 步未成功

---

## 🧪 使用示例

### 示例 1：手动控制环境

```python
from grasp_env import GraspEnv

# 创建环境
env = GraspEnv(render_mode='human')

# 重置
state, info = env.reset()
print(f"物体位置：{info['object_pos']}")
print(f"物体类别：{info['object_class']}")

# 执行动作
for step in range(20):
    action = env.action_space.sample()  # 随机动作
    next_state, reward, terminated, truncated, info = env.step(action)
    
    if info.get('grasped', False):
        print(f"第{step}步成功抓取！")
        break

env.close()
```

### 示例 2：使用训练好的智能体

```python
from grasp_env import DQNAgent, GraspEnv

# 创建智能体
agent = DQNAgent()

# 加载训练好的模型
agent.load('dqn_grasp_model.pth')

# 测试
env = GraspEnv(render_mode='human')
state, _ = env.reset()
done = False

while not done:
    # 测试模式（不使用ε-greedy）
    action = agent.select_action(state, training=False)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

print(f"成功：{info.get('grasped', False)}")
env.close()
```

### 示例 3：自定义训练

```python
from grasp_env import GraspEnv, DQNAgent
import numpy as np

env = GraspEnv()
agent = DQNAgent(
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995
)

for episode in range(1000):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.train_step()
        
        state = next_state
        episode_reward += reward
    
    if episode % 100 == 0:
        print(f"Episode {episode}: Reward={episode_reward:.2f}")

# 保存模型
agent.save('my_custom_model.pth')
```

---

## 📊 训练曲线示例

典型训练过程（500 回合）：

```
Episode 50/500 | Avg Reward: 2.34 | Success Rate: 23.0% | Epsilon: 0.78
Episode 100/500 | Avg Reward: 5.67 | Success Rate: 56.0% | Epsilon: 0.61
Episode 150/500 | Avg Reward: 7.89 | Success Rate: 78.0% | Epsilon: 0.47
Episode 200/500 | Avg Reward: 8.92 | Success Rate: 89.0% | Epsilon: 0.37
Episode 250/500 | Avg Reward: 9.45 | Success Rate: 94.0% | Epsilon: 0.29
...
Episode 500/500 | Avg Reward: 9.87 | Success Rate: 98.0% | Epsilon: 0.09
```

---

## 🔧 扩展建议

### 1. 添加更复杂的场景

```python
# 在 GraspEnv 中添加障碍物
class AdvancedGraspEnv(GraspEnv):
    def __init__(self, num_obstacles=3, **kwargs):
        super().__init__(**kwargs)
        self.obstacles = []  # 障碍物位置列表
        # ... 实现碰撞检测
```

### 2. 支持连续动作空间

```python
# 将动作空间改为连续（用于 PPO、SAC 等算法）
from gymnasium import spaces
self.action_space = spaces.Box(
    low=np.array([-1, -1, 0]),  # [dx, dy, gripper]
    high=np.array([1, 1, 1]),
    dtype=np.float32
)
```

### 3. 添加视觉输入

```python
# 添加 RGB 图像观测
from gymnasium import spaces
self.observation_space = spaces.Dict({
    'image': spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
    'state': spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32)
})
```

---

## 📚 相关资源

- **VLA-Grasp 论文**：见 `paper_outline.md` 和 `paper_draft.md`
- **Gymnasium 文档**：https://gymnasium.farama.org/
- **PyTorch 教程**：https://pytorch.org/tutorials/
- **DQN 原论文**：Mnih et al., "Playing Atari with Deep Reinforcement Learning", 2013

---

## ⚠️ 注意事项

1. **这是简化环境**：真实机器人实验请使用 Isaac Sim、MuJoCo 或 PyBullet
2. **2D 平面假设**：实际抓取是 3D 问题，本环境仅用于算法验证
3. **无物理引擎**：碰撞检测为简化版本，无真实物理模拟
4. **单物体场景**：实际场景可能有多个物体和遮挡

---

## 📝 许可证

MIT License - 可自由用于研究和教学

---

## 🙏 致谢

- Gymnasium 团队提供的环境框架
- PyTorch 团队的深度学习库
- VLA-Grasp 研究项目支持

---

**最后更新**：2026-03-06  
**联系方式**：[待填写]
