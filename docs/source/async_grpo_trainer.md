# Async GRPO

> [!WARNING]
> This trainer is **experimental** and may change or be removed in any release without prior deprecation.

## Overview

[`AsyncGRPOTrainer`] decouples rollout generation from training: a background worker continuously streams completions from a vLLM server while the training loop consumes them, keeping the GPU busy at all times. For a fully-featured trainer, use [`GRPOTrainer`] instead.

## Aim

This trainer is intentionally kept minimal and is not meant to grow into a general-purpose solution. If you need a feature that is not supported, we recommend cloning the repository and adapting the trainer to your needs directly. New features will only be considered when there is significant community demand.

## AsyncGRPOConfig

[[autodoc]] trl.experimental.async_grpo.AsyncGRPOConfig

## AsyncGRPOTrainer

[[autodoc]] trl.experimental.async_grpo.AsyncGRPOTrainer
