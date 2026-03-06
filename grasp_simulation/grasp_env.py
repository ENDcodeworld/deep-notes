"""
GraspEnv: 机器人抓取仿真环境
基于 PyTorch + Gym 的简化抓取仿真环境，用于 DQN 算法训练

作者：志哥的研究团队
日期：2026-03-06
用途：VLA-Grasp 论文基线实验
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspEnv(gym.Env):
    """
    简化的 2D 平面抓取仿真环境
    
    状态空间：
        - 物体位置 (x, y): 2 维
        - 物体类别 one-hot: 5 维
        - 机械臂位置 (x, y): 2 维
        - 夹爪开合状态：1 维
        总计：10 维
    
    动作空间：
        - 离散动作：0-停止，1-左移，2-右移，3-上移，4-下移，5-闭合夹爪，6-打开夹爪
        总计：7 个离散动作
    
    奖励设计：
        - 成功抓取：+10
        - 接近物体：+0.1（每步）
        - 无效动作：-0.01
        - 碰撞边界：-1
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        num_objects: int = 5,
        workspace_size: float = 1.0,
        max_steps: int = 100,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # 环境参数
        self.num_objects = num_objects  # 物体类别数
        self.workspace_size = workspace_size  # 工作空间大小（米）
        self.max_steps = max_steps  # 最大步数
        self.render_mode = render_mode
        
        # 状态空间：10 维连续向量
        # [物体 x, 物体 y, 物体类别 (one-hot 5 维), 机械臂 x, 机械臂 y, 夹爪状态]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )
        
        # 动作空间：7 个离散动作
        self.action_space = spaces.Discrete(7)
        
        # 环境状态变量
        self.object_pos = None  # 物体位置
        self.object_class = None  # 物体类别
        self.arm_pos = None  # 机械臂位置
        self.gripper_open = None  # 夹爪状态（True=打开，False=闭合）
        self.current_step = None  # 当前步数
        self.grasped = None  # 是否已抓取
        
        # 渲染相关
        self.viewer = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机初始化物体位置（在工作空间内）
        self.object_pos = self.np_random.uniform(
            low=-self.workspace_size/2,
            high=self.workspace_size/2,
            size=(2,)
        )
        
        # 随机选择物体类别
        self.object_class = self.np_random.integers(0, self.num_objects)
        
        # 随机初始化机械臂位置（在工作空间边缘）
        self.arm_pos = self.np_random.uniform(
            low=-self.workspace_size/2,
            high=self.workspace_size/2,
            size=(2,)
        )
        
        # 夹爪初始为打开状态
        self.gripper_open = True
        
        # 重置计数器和状态
        self.current_step = 0
        self.grasped = False
        
        # 构建观测向量
        obs = self._get_observation()
        
        # 信息字典
        info = {
            'object_pos': self.object_pos.copy(),
            'object_class': self.object_class,
            'arm_pos': self.arm_pos.copy()
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作
        
        参数：
            action: 0-6 的整数，分别对应不同动作
            
        返回：
            observation: 新观测
            reward: 奖励
            terminated: 是否终止（成功或失败）
            truncated: 是否超时
            info: 信息字典
        """
        # 检查动作合法性
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # 执行动作
        self._execute_action(action)
        
        # 步数 +1
        self.current_step += 1
        
        # 计算奖励
        reward = self._calculate_reward(action)
        
        # 检查是否终止
        terminated = self.grasped  # 成功抓取则终止
        
        # 检查是否超时
        truncated = self.current_step >= self.max_steps
        
        # 获取新观测
        obs = self._get_observation()
        
        # 信息字典
        info = {
            'grasped': self.grasped,
            'steps': self.current_step,
            'distance': np.linalg.norm(self.arm_pos - self.object_pos)
        }
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: int):
        """执行动作"""
        # 动作映射
        # 0: 停止
        # 1: 左移 (-x)
        # 2: 右移 (+x)
        # 3: 上移 (+y)
        # 4: 下移 (-y)
        # 5: 闭合夹爪
        # 6: 打开夹爪
        
        move_step = 0.05  # 每步移动距离（米）
        
        if action == 1:  # 左移
            self.arm_pos[0] -= move_step
        elif action == 2:  # 右移
            self.arm_pos[0] += move_step
        elif action == 3:  # 上移
            self.arm_pos[1] += move_step
        elif action == 4:  # 下移
            self.arm_pos[1] -= move_step
        elif action == 5:  # 闭合夹爪
            if self.gripper_open:  # 只有打开时才能闭合
                self.gripper_open = False
                # 检查是否抓取成功
                distance = np.linalg.norm(self.arm_pos - self.object_pos)
                if distance < 0.1:  # 距离小于 10cm 算成功
                    self.grasped = True
        elif action == 6:  # 打开夹爪
            self.gripper_open = True
        
        # 边界检查（防止移出工作空间）
        self.arm_pos = np.clip(
            self.arm_pos,
            -self.workspace_size/2,
            self.workspace_size/2
        )
    
    def _calculate_reward(self, action: int) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 成功抓取：大奖励
        if self.grasped:
            reward += 10.0
        
        # 接近物体：小奖励（鼓励靠近）
        distance = np.linalg.norm(self.arm_pos - self.object_pos)
        if distance < 0.2:  # 20cm 以内
            reward += 0.1 * (1 - distance / 0.2)
        
        # 无效动作：小惩罚
        if action == 0:  # 停止
            reward -= 0.01
        
        # 碰撞边界：惩罚
        if (np.abs(self.arm_pos[0]) >= self.workspace_size/2 - 0.01 or
            np.abs(self.arm_pos[1]) >= self.workspace_size/2 - 0.01):
            reward -= 1.0
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """构建观测向量"""
        # 物体位置（2 维）
        obs_parts = [self.object_pos]
        
        # 物体类别 one-hot 编码（5 维）
        class_onehot = np.zeros(self.num_objects, dtype=np.float32)
        class_onehot[self.object_class] = 1.0
        obs_parts.append(class_onehot)
        
        # 机械臂位置（2 维）
        obs_parts.append(self.arm_pos)
        
        # 夹爪状态（1 维）
        obs_parts.append(np.array([1.0 if self.gripper_open else 0.0], dtype=np.float32))
        
        # 拼接所有部分
        obs = np.concatenate(obs_parts).astype(np.float32)
        
        return obs
    
    def render(self):
        """渲染环境（简化版，打印状态）"""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Object: pos={self.object_pos}, class={self.object_class}")
            print(f"Arm: pos={self.arm_pos}, gripper={'open' if self.gripper_open else 'closed'}")
            print(f"Grasped: {self.grasped}")
            print("-" * 40)
    
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ============================================================================
# DQN 算法实现
# ============================================================================

class DQN(nn.Module):
    """
    DQN 网络：简单的全连接网络
    
    输入：10 维观测向量
    输出：7 个动作的 Q 值
    """
    
    def __init__(self, input_dim: int = 10, output_dim: int = 7, hidden_dim: int = 128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)


class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储 (state, action, reward, next_state, done) 元组
    支持随机采样
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """随机采样一批经验"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN 智能体
    
    包含：
        - Q 网络
        - 目标网络
        - 经验回放
        - ε-greedy 策略
    """
    
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 7,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10
    ):
        # 超参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 网络
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 训练计数
        self.steps = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（ε-greedy 策略）
        
        参数：
            state: 当前状态
            training: 是否训练模式（训练时使用ε-greedy，测试时纯贪婪）
            
        返回：
            action: 选择的动作
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机动作
            return np.random.randint(self.action_dim)
        else:
            # 利用：选择 Q 值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self) -> float:
        """
        执行一步训练
        
        返回：
            loss: 当前训练损失
        """
        # 检查缓冲区是否有足够数据
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 采样一批经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为 tensor
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算当前 Q 值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标 Q 值（Bellman 方程）
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失（MSE）
        loss = F.mse_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新计数
        self.steps += 1
        
        # 定期更新目标网络
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减ε
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


# ============================================================================
# 训练脚本
# ============================================================================

def train_dqn(num_episodes: int = 500, render: bool = False):
    """
    训练 DQN 智能体
    
    参数：
        num_episodes: 训练回合数
        render: 是否渲染
    """
    # 创建环境
    env = GraspEnv(render_mode='human' if render else None)
    
    # 创建智能体
    agent = DQNAgent()
    
    # 训练统计
    episode_rewards = []
    success_count = 0
    
    print("开始训练 DQN...")
    print(f"环境：{env.observation_space.shape[0]}维状态，{env.action_space.n}个动作")
    print(f"训练回合：{num_episodes}")
    print("-" * 40)
    
    for episode in range(num_episodes):
        # 重置环境
        state, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 训练一步
            loss = agent.train_step()
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            # 渲染（可选）
            if render and episode % 50 == 0:
                env.render()
        
        # 记录统计
        episode_rewards.append(episode_reward)
        if info.get('grasped', False):
            success_count += 1
        
        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            success_rate = success_count / (episode + 1) * 100
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {success_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # 训练完成
    print("-" * 40)
    print("训练完成！")
    print(f"最终成功率：{success_count / num_episodes * 100:.1f}%")
    print(f"平均奖励：{np.mean(episode_rewards[-100:]):.2f}")
    
    # 保存模型
    agent.save('dqn_grasp_model.pth')
    print("模型已保存到：dqn_grasp_model.pth")
    
    # 关闭环境
    env.close()
    
    return agent, episode_rewards


if __name__ == '__main__':
    # 训练示例
    agent, rewards = train_dqn(num_episodes=500, render=False)
    
    # 测试训练好的模型
    print("\n" + "=" * 40)
    print("测试训练好的模型")
    print("=" * 40)
    
    env = GraspEnv(render_mode='human')
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 测试时不使用ε-greedy
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
    
    print(f"测试完成 | 总奖励：{total_reward:.2f} | 成功：{info.get('grasped', False)}")
    env.close()
