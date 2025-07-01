import torch

tensor_random = torch.rand(3, 4)
tensor_zeros = torch.zeros(2, 3, 4)
tensor_ones = torch.ones(5, 5)
tensor_arange = torch.arange(16).reshape(4, 4)

a = torch.rand(3, 4)
b = torch.rand(4, 3)
a_t = a.T
matmul_result = (a @ b)
b_t = b.T
elemwise_mul = (a  * b_t)
sum_a = a.sum()

tensor = torch.rand(5, 5, 5)
first_row = tensor[0]
last_column = tensor[:, :, -1]
center_submatrix = tensor[2:4, 2:4, 2:4]
even_elements = tensor[::2, ::2, ::2]

tensor1 = torch.arange(24)
tensor_2x12 = tensor1.reshape(2, 12)
tensor_3x8 = tensor1.reshape(3, 8)
tensor_4x6 = tensor1.reshape(4, 6)
tensor_2x3x4 = tensor1.reshape(2, 3, 4)
tensor_2x2x2x3 = tensor1.reshape(2, 2, 2, 3)

