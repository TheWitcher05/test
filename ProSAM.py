import numpy as np
import torch  
import torch.nn as nn  
import torch.optim as optim  
import copy
from tensorboardX import SummaryWriter
torch.manual_seed(42) 
import tensorflow as tf  

d=20
std_dev=0.1
sigma=100
kappa=1
sample_num=7000
test_num=300





def generate_positive_definite_matrix_with_condition_number_torch(condition_number, size):  
    # 生成一个随机的正交矩阵 U 和 V  
    U, _ = torch.qr(torch.randn(size, size))  
    V, _ = torch.qr(torch.randn(size, size))  
  
    # 生成一个对角矩阵 Sigma，对角线上的元素为奇异值  
    singular_values = torch.linspace(1, condition_number, steps=size)  
  
    # 生成矩阵 A = U * Sigma * V^T  
    A = torch.matmul(U, torch.matmul(torch.diag(singular_values), V.t()))  
  
    # 生成正定矩阵  
    positive_definite_matrix = torch.matmul(A, A.t())  
  
    return positive_definite_matrix
  
# 生成条件数为 100 的 3x3 正定矩阵  
condition_number = kappa 
matrix_size = d 
cov_matrix = generate_positive_definite_matrix_with_condition_number_torch(condition_number, matrix_size)  



logdir = "./logs" 
file_writer = tf.summary.create_file_writer(logdir) 
writer=SummaryWriter('./logs')
# # Loop from 0 to 199 and get the sine value of each number 
# for i in range(200): 
#     with file_writer.as_default(): 
#         tf.summary.scalar('sine wave', np.math.sin(i), step=i)

# 定义协方差矩阵  
# cov_matrix =kappa*torch.eye(d,d)
# cov_matrix = torch.mm(cov_matrix, cov_matrix.t())  
  
# 从多变量高斯分布中采样  
mean = torch.zeros(d)  # 均值  
mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_matrix)  
samples = mvn.sample((sample_num,))  
test_samples=mvn.sample((test_num,))  


semi_matrix=torch.randn(d, d)
semi_matrix=sigma*torch.mm(semi_matrix, semi_matrix.t())
truth_matrix =  torch.eye(d)+ semi_matrix
label_list=[]
for i in samples:
    # 生成零均值的 d 维高斯随机向量  
    mean = torch.zeros(d)  # 均值为 0  
  
    gaussian_vector = mean + std_dev * torch.randn(d)  # 生成高斯随机向量 
    label=torch.matmul(i,truth_matrix)+gaussian_vector
    label_list.append(  label)
label_torch = torch.stack( label_list)   
 

test_list=[]
for i in test_samples:
    mean = torch.zeros(d) 
  
    gaussian_vector = mean + std_dev * torch.randn(d)  # 生成高斯随机向量 
    label=torch.matmul(i,truth_matrix)+gaussian_vector
    test_list.append(  label)
test_torch = torch.stack( test_list)   


# update_rule="worst"
# update_rule="SAM"
# update_rule="Nesterov"
# update_rule="GD"
# update_rule="Random"
# update_rule="Vanilla_GD"
# update_rule="Vanilla_Nesterov"
# update_rule="Stacking"

update_rule="Vanilla_Worst"
# update_rule="Vanilla_Random"
# update_rule="Vanilla_SAM"
criterion = nn.MSELoss()
learning_rate = 1/(torch.norm(cov_matrix)) 
parameter_matrix = torch.randn(d, d, requires_grad=True)
previous_matrix=parameter_matrix.clone()
# previous_matrix=copy.deepcopy(parameter_matrix)
model=[parameter_matrix]
optimizer = optim.SGD(model, lr=learning_rate)  

num_epochs =30

parameters_list=[]
for epoch in range(num_epochs):  
    # 前向传播  
    parameters_list.append(parameter_matrix)
  
 

    if(update_rule=="GD"):

        output = torch.matmul(samples,parameter_matrix) 
        loss = criterion(output, label_torch)  
        optimizer.zero_grad()   
        loss.backward()  
        optimizer.step() 

    

    if(update_rule=="SAM"):
        output = torch.matmul(samples,parameter_matrix)
    
        loss = criterion(output, label_torch)  

        optimizer.zero_grad()  
        loss.backward() 
        gradients =parameter_matrix.grad
        grad_norm=rho*(gradients/torch.norm(gradients)) 
        loss_new=criterion(label,torch.matmul(samples,torch.add(parameter_matrix,grad_norm)))
        optimizer.zero_grad()
        loss_new.backward()
        optimizer.step()
    
    if(update_rule=="worst"):
        output = torch.matmul(samples,parameter_matrix)

        loss = criterion(output, label_torch)  

        optimizer.zero_grad()  
        loss.backward()  
        gradients =parameter_matrix.grad
        grad_norm=rho*(gradients/torch.norm(gradients)) 
        loss_new=criterion(label,torch.matmul(samples,torch.add(parameter_matrix,grad_norm)))
        optimizer.zero_grad()
        loss_new.backward()
        parameter_matrix.grad=torch.add(  parameter_matrix.grad,    grad_norm)
        optimizer.step()

    if(update_rule=="Vanilla_SAM"):
        smooth_value=torch.norm(cov_matrix)
        rho=0.5
        W_hat=torch.sub(parameter_matrix,truth_matrix)
        loss=0.5*torch.trace(torch.matmul(torch.matmul(  W_hat,cov_matrix),W_hat.t()))
        optimizer.zero_grad()
        loss.backward()
        grad_norm=parameter_matrix.grad/(torch.norm(parameter_matrix.grad))
        W_new=torch.sub(      torch.add(parameter_matrix,     rho*grad_norm),truth_matrix)
        loss_new=0.5*torch.trace(torch.matmul(torch.matmul(W_new,cov_matrix),W_new.t()))
        optimizer.zero_grad()
        loss_new.backward()
        # parameter_matrix.grad=torch.add(  parameter_matrix.grad, rho*grad_norm)
        optimizer.step()
        output = torch.matmul(samples,parameter_matrix) 
        loss = criterion(output, label_torch)  


    if(update_rule=="Vanilla_Worst"):
        smooth_value=torch.norm(cov_matrix)
        rho=0.5
        W_hat=torch.sub(parameter_matrix,truth_matrix)
        loss=0.5*torch.trace(torch.matmul(torch.matmul(  W_hat,cov_matrix),W_hat.t()))
        optimizer.zero_grad()
        loss.backward()
        grad_norm=parameter_matrix.grad/(torch.norm(parameter_matrix.grad))
        W_new=torch.sub(      torch.add(parameter_matrix,     rho*grad_norm),truth_matrix)
        loss_new=0.5*torch.trace(torch.matmul(torch.matmul(W_new,cov_matrix),W_new.t()))
        optimizer.zero_grad()
        loss_new.backward()
        parameter_matrix.grad=torch.add(  parameter_matrix.grad, rho*grad_norm)
        optimizer.step()
        output = torch.matmul(samples,parameter_matrix) 
        loss = criterion(output, label_torch)  

    if(update_rule=="Vanilla_GD"):
        W_hat=torch.sub(parameter_matrix,truth_matrix)
        loss=0.5*torch.trace(torch.matmul(torch.matmul(  W_hat,cov_matrix),W_hat.t()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = torch.matmul(samples,parameter_matrix) 
        loss = criterion(output, label_torch)  

    if(update_rule=="Vanilla_Random"):
        beta=1
        momentum=torch.randn(d,d)
       
        W_hat=torch.sub(torch.add(parameter_matrix,beta*momentum),truth_matrix)
        loss=0.5*torch.trace(torch.matmul(torch.matmul(  W_hat,cov_matrix),W_hat.t()))
        optimizer.zero_grad()
        loss.backward()
        parameter_matrix.grad=torch.add(  parameter_matrix.grad, beta*momentum)
        optimizer.step()
        output = torch.matmul(samples,parameter_matrix) 
        loss = criterion(output, label_torch)  

    if(update_rule=="Vanilla_Nesterov"):
        beta=0.1
        momentum=torch.sub(parameter_matrix,previous_matrix)
        # grad=0.01*(torch.matmul(torch.sub(parameter_matrix,truth_matrix),cov_matrix))
        previous_matrix=parameter_matrix.clone()
        W_hat=torch.sub(torch.add(parameter_matrix,beta*momentum),truth_matrix)
        loss=0.5*torch.trace(torch.matmul(torch.matmul(  W_hat,cov_matrix),W_hat.t()))
        optimizer.zero_grad()
        loss.backward()
        parameter_matrix.grad=torch.add(  parameter_matrix.grad, beta*momentum)
        optimizer.step()
        output = torch.matmul(samples,parameter_matrix) 
        loss = criterion(output, label_torch)  

    if(update_rule=="Stacking"):
        beta=0.1
        momentum=torch.sub(parameter_matrix,previous_matrix)
        # grad=0.01*(torch.matmul(torch.sub(parameter_matrix,truth_matrix),cov_matrix))
        previous_matrix=parameter_matrix.clone()
        W_I=torch.matmul(torch.inverse( previous_matrix),parameter_matrix)
        W_hat=torch.sub(torch.add(parameter_matrix,beta*torch.matmul(W_I,momentum)),truth_matrix)
        loss=0.5*torch.trace(torch.matmul(torch.matmul(  W_hat,cov_matrix),W_hat.t()))
        optimizer.zero_grad()
        loss.backward()
        parameter_matrix.grad=torch.add(  parameter_matrix.grad, beta*momentum)
        optimizer.step()
        output = torch.matmul(samples,parameter_matrix) 
        loss = criterion(output, label_torch)  






  

    test_loss= criterion(torch.matmul(test_samples,parameter_matrix), test_torch)  
    # writer.add_scalars('std=_Training_loss',{update_rule:    loss.data.item()},epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {   test_loss.item():.4f}')  
   
    print("_____________________")
      
