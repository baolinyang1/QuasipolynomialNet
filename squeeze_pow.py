import torch

def power_matrix_operation(A, B):
    # Expand dimensions of A and B for broadcasting
    A_expanded = A.unsqueeze(1)  # shape: (n, 1, m)
    # B_expanded = B.unsqueeze(0)  # shape: (1, p, m)
    
    # Perform element-wise exponentiation
    C = A_expanded ** B  # shape: (n, p, m)
    
    return C

# Example sizes for A and B
n, m, p = 4, 2, 2

# Create random matrices A and B
A = torch.ones(n, m) * 2
# B = torch.arange(p, m)
B = torch.arange(1, p * m + 1).reshape(p, m)

print("Matrix B:"+str(B))
# Perform the operation
C = power_matrix_operation(A, B)

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nResulting Matrix C:")
print(C)
