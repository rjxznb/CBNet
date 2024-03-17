import torch
import torch.nn as nn
import torch.optim as optim

# 假设的模型、损失函数和优化器初始化
model = YourModel()  # 替换为你的模型
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 任务损失函数
loss_fn_task1 = nn.MSELoss()
loss_fn_task2 = nn.CrossEntropyLoss()

# 初始化任务权重，这里简单地设置为1
weights = torch.tensor([1.0, 1.0], requires_grad=True)
alpha = 0.16  # GradNorm算法的超参数

# 训练循环
for data in dataloader:  # 假设dataloader已经准备好
    optimizer.zero_grad()

    # 前向传播
    output_task1, output_task2 = model(data)

    # 计算原始任务损失
    loss_task1 = loss_fn_task1(output_task1, target_task1)
    loss_task2 = loss_fn_task2(output_task2, target_task2)

    # 计算加权损失
    weighted_loss = weights[0] * loss_task1 + weights[1] * loss_task2
    weighted_loss.backward(retain_graph=True)  # 保留计算图以用于GradNorm更新

    # GradNorm权重更新
    W_grads = torch.autograd.grad(weighted_loss, weights, retain_graph=True)
    W_grad_norms = torch.norm(torch.stack(W_grads), 2, 1)
    loss_ratios = torch.tensor([loss_task1.item() / loss_task2.item()], requires_grad=False)
    loss_ratio_grads = loss_ratios ** alpha
    target_grad_norms = W_grad_norms * loss_ratio_grads
    grad_norms_ratio = target_grad_norms / W_grad_norms
    weights.grad = W_grads[0] * grad_norms_ratio[0]  # 这里简化处理，实际使用中可能需要更复杂的操作

    # 使用优化器更新模型参数
    optimizer.step()

    # 更新任务权重（确保权重为正且进行适当的归一化或限制）
    with torch.no_grad():
        weights += learning_rate * weights.grad  # 更新权重，这里的learning_rate需要根据实际情况调整
        weights = torch.clamp(weights, min=0.01)  # 防止权重过小;





















