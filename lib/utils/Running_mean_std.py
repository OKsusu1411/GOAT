import numpy as np


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count



import torch
import torch.nn as nn

class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = (), device: str = "cuda:0"):
        """
        Calculates the running mean and std of a data stream using PyTorch
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        :param device: "cpu" or "cuda"
        """
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        
        # 모델 저장/로드 시 상태가 유지되도록 register_buffer 사용 (학습 파라미터 X)
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32, device=device))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32, device=device))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32, device=device))

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape, epsilon=self.epsilon, device=self.device)
        new_object.mean.copy_(self.mean)
        new_object.var.copy_(self.var)
        new_object.count.copy_(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, x: torch.Tensor) -> None:
        """
        Update the statistics with a new batch of data.
        :param x: Input tensor (batch_size, dims)
        """
        # 입력 데이터가 텐서가 아니면 변환
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        batch_mean = torch.mean(x, dim=0)
        # np.var는 기본적으로 unbiased=False (N으로 나눔)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32, device=self.device)

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: torch.Tensor) -> None:
        """
        Updates internal stats based on external batch moments.
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        
        # np.square -> torch.square or ** 2
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        """
        Optional: Applies normalization to x.
        """
        if train:
            self.update(x)
        
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)