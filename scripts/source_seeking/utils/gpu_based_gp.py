import numpy as np
from sklearn.utils import check_random_state
import torch
import gpytorch
from scipy.ndimage import maximum_filter


        
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.utils import check_random_state
import time

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, sigma_f=1.0, ell=1.0, sigma_n=0.1):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # RBF kernel with amplitude (sigma_f) and lengthscale (ell)
        # 添加与 CPU 版本相同的约束 (0.5, 2.0)
        lengthscale_constraint = gpytorch.constraints.Interval(0.5, 2.0)
        base_kernel = gpytorch.kernels.RBFKernel(lengthscale_constraint=lengthscale_constraint)
        base_kernel.lengthscale = ell
        
        # 设置 outputscale (sigma_f²) 的约束
        outputscale_constraint = gpytorch.constraints.Positive()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_constraint=outputscale_constraint
        )
        self.covar_module.outputscale = sigma_f**2

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
class TorchGPModel():
    def __init__(self, sigma_f=1.0, ell=1.0, sigma_n=0.1):
        # 添加噪声约束，与 CPU 版本保持一致
        noise_constraint = gpytorch.constraints.Interval(1e-5, 1.0)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint
        )
        self.likelihood.noise = sigma_n**2
        
        # Use dummy data for initial model creation
        dummy_x = torch.zeros(1, 1)
        dummy_y = torch.zeros(1)
        self.model = ExactGPModel(dummy_x, dummy_y, self.likelihood, 
                                sigma_f=sigma_f, ell=ell, sigma_n=sigma_n)
        
    def fit(self, X, Y, num_iterations=100):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, np.ndarray):
            X = torch.tensor(X).float()
        if isinstance(Y, np.ndarray):
            Y = torch.tensor(Y).float()
        if len(X.shape) == 2:
            X = X
        if len(Y.shape) == 2:
            Y = torch.reshape(Y, [-1, ])
            
        self.model.set_train_data(X, Y, strict=False)
        
        self.model.train()
        self.likelihood.train()
        
        # 优化设置
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=0.1)
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # 添加早停机制
        best_loss = float('inf')
        patience = 10
        no_improve = 0
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            output = self.model(self.model.train_inputs[0])
            loss = -mll(output, self.model.train_targets)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve > patience:
                break

    def predict(self, X, return_std=False, return_cov=False, return_tensor=False):
        self.model.eval()
        self.likelihood.eval()
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, np.ndarray):
            X = torch.tensor(X).float()
        if len(X.shape) == 1:
            X = torch.reshape(X, [1, -1])
        with gpytorch.settings.fast_pred_var():
            f_pred = self.model(X)
            if return_tensor:
                if return_std:
                    return f_pred.mean, f_pred.variance
                elif return_cov:
                    return f_pred.mean, f_pred.covariance_matrix
                else:
                    return f_pred.mean
            else:
                if return_std:
                    return f_pred.mean.detach().numpy(), f_pred.variance.detach().numpy()
                elif return_cov:
                    return f_pred.mean.detach().numpy(), f_pred.covariance_matrix.detach().numpy()
                else:
                    return f_pred.mean.detach().numpy()

    def sample_y(self, X, n_samples, random_state=None):
        rng = check_random_state(random_state)

        y_mean, y_cov = self.predict(X, return_cov=True)
        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = [
                rng.multivariate_normal(
                    y_mean[:, target], y_cov[..., target], n_samples
                ).T[:, np.newaxis]
                for target in range(y_mean.shape[1])
            ]
            y_samples = np.hstack(y_samples)
        return y_samples

    @property
    def y_train_(self):
        return self.model.train_targets.detach().numpy()

    @property
    def X_train_(self):
        return self.model.train_inputs[0].detach().numpy()
        
    def get_kernel_params(self):
        """获取当前kernel参数"""
        return {
            'sigma_f**2': self.model.covar_module.outputscale.item(),
            'ell': self.model.covar_module.base_kernel.lengthscale.item(),
            'sigma_n**2': self.likelihood.noise.item()
        }


class CpuGPModel:
    def __init__(self, sigma_f=1.0, ell=1.0, sigma_n=0.1):
        self.kernel = sigma_f**2 * RBF(length_scale=ell, length_scale_bounds=(0.5, 2)) + \
                     WhiteKernel(noise_level=sigma_n)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=10
        )
        
    def fit(self, X, y):
        if len(y.shape) == 2:
            y = y.ravel()
        self.gp.fit(X, y)
        
    def predict(self, X, return_std=False, return_cov=False):
        if return_cov:
            return self.gp.predict(X, return_cov=True)
        elif return_std:
            return self.gp.predict(X, return_std=True)
        else:
            return self.gp.predict(X)
            
    def sample_y(self, X, n_samples, random_state=None):
        return self.gp.sample_y(X, n_samples, random_state)
    
    @property
    def X_train_(self):
        return self.gp.X_train_
    
    @property
    def y_train_(self):
        return self.gp.y_train_
        
    def get_kernel_params(self):
        """获取kernel参数"""
        # 分解kernel
        k1, k2 = self.gp.kernel_.k1, self.gp.kernel_.k2  # k1 是 sigma_f^2 * RBF, k2 是 WhiteKernel
        k1_params = k1.get_params()
        # RBF kernel 的参数在 k1 的第二个部分
        rbf_kernel = k1_params['k2']
        
        return {
            'sigma_f**2': k1_params['k1'],  # constant value is the first part of k1
            'ell': rbf_kernel.length_scale,
            'sigma_n**2': k2.noise_level
        }

# 比较测试
def compare_models():
    # 生成测试数据
    np.random.seed(0)
    X_train = np.random.rand(100, 1) * 10
    y_train = np.sin(X_train.ravel()) + np.random.normal(0, 0.1, 100)
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    
    # 初始化模型
    torch_gp = TorchGPModel(sigma_f=1.0, ell=1.0, sigma_n=0.1)
    cpu_gp = CpuGPModel(sigma_f=1.0, ell=1.0, sigma_n=0.1)
    
    # 训练和预测时间比较
    print("Training comparison:")
    # PyTorch版本
    start_time = time.time()
    torch_gp.fit(X_train, y_train, num_iterations=100)
    torch_time_train = time.time() - start_time
    print(f"TorchGP training time: {torch_time_train:.4f}s")
    
    # CPU版本
    start_time = time.time()
    cpu_gp.fit(X_train, y_train)
    cpu_time_train = time.time() - start_time
    print(f"CpuGP training time: {cpu_time_train:.4f}s")
    
    print("\nPrediction comparison:")
    # PyTorch版本预测
    start_time = time.time()
    y_pred_torch, y_std_torch = torch_gp.predict(X_test, return_std=True)
    torch_time_pred = time.time() - start_time
    print(f"TorchGP prediction time: {torch_time_pred:.4f}s")
    
    # CPU版本预测
    start_time = time.time()
    y_pred_cpu, y_std_cpu = cpu_gp.predict(X_test, return_std=True)
    cpu_time_pred = time.time() - start_time
    print(f"CpuGP prediction time: {cpu_time_pred:.4f}s")
    
    print("\nKernel parameters after training:")
    print("TorchGP:", torch_gp.get_kernel_params())
    print("CpuGP:", cpu_gp.get_kernel_params())
    
    # 绘制结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    
    # 训练数据
    plt.scatter(X_train, y_train, c='black', label='Training points')
    
    # TorchGP预测
    plt.plot(X_test, y_pred_torch, 'r-', label='TorchGP prediction')
    plt.fill_between(X_test.ravel(), 
                    y_pred_torch - 2*y_std_torch,
                    y_pred_torch + 2*y_std_torch, 
                    color='red', alpha=0.2)
    
    # CpuGP预测
    plt.plot(X_test, y_pred_cpu, 'b--', label='CpuGP prediction')
    plt.fill_between(X_test.ravel(), 
                    y_pred_cpu - 2*y_std_cpu,
                    y_pred_cpu + 2*y_std_cpu, 
                    color='blue', alpha=0.2)
    
    plt.legend()
    plt.title('Comparison of TorchGP and CpuGP')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

# 运行比较
if __name__ == "__main__":
    compare_models()