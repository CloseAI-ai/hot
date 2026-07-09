# HOT: Harmonic Oscillator Transformer

> 基于耦合谐振子动力学的时序建模新范式

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## 概述

HOT（Harmonic Oscillator Transformer）是一种摒弃静态位置嵌入的替代架构：将每个 Token 视为具有内在固有频率的耦合谐振子，其状态由复数域相位角表征；注意力权重通过内容自适应的相位同步门控机制动态生成，使 Token 间的信息通路由语义节律的相干性决定。

## 核心特性

- **动态位置编码**：摒弃静态位置嵌入，位置关系由内容驱动的语义节律动态涌现
- **相位同步门控**：注意力权重通过内容自适应的相位同步机制生成
- **长度外推能力**：在不引入线性注意力或状态空间近似的前提下，为长度外推提供新可能性
- **兼容现有范式**：可与 FlashAttention 等现有计算范式协同工作

## 文档

- [研究论文](article.md) — 详述理论框架、工程实施策略及收敛性保障机制
- [开发计划](dev-plan.md) — 125M 参数规模的研究原型实现计划

## 许可证

本项目基于 [GNU Affero General Public License v3.0](LICENSE) 发布。

## 组织

本项目由 [CloseAI.ai](https://github.com/CloseAI-ai) 维护。
