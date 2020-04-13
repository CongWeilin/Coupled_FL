import matplotlib.pyplot as plt
import pickle
import numpy as np

fig, axs = plt.subplots()

"""
Coupled_FL
"""
# diversity 0
with open('coupled_0.00.pkl', 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(train_loss), step=20)
x = np.arange(len(select))
y = [train_loss[i] for i in select]
axs.plot(x,y,label='Diversity 0 (Coupled_FL)', linestyle='-')

# diversity 0.5
with open('coupled_0.50.pkl', 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(train_loss), step=20)
x = np.arange(len(select))
y = [train_loss[i] for i in select]
axs.plot(x,y,label='Diversity 0.5 (Coupled_FL)', linestyle='-')

# diversity 1
with open('coupled_1.00.pkl', 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(train_loss), step=20)
x = np.arange(len(select))
y = [train_loss[i] for i in select]
axs.plot(x,y,label='Diversity 1 (Coupled_FL)', linestyle='-')

"""
FedAvg
"""
# diversity 0
with open('fedavg_0.00.pkl', 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(train_loss), step=20)
x = np.arange(len(select))
y = [train_loss[i] for i in select]
axs.plot(x,y,label='Diversity 0 (FedAvg)', linestyle='--')

# diversity 0.5
with open('fedavg_0.50.pkl', 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(train_loss), step=20)
x = np.arange(len(select))
y = [train_loss[i] for i in select]
axs.plot(x,y,label='Diversity 0.5 (FedAvg)', linestyle='--')

# diversity 1
with open('fedavg_1.00.pkl', 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(train_loss), step=20)
x = np.arange(len(select))
y = [train_loss[i] for i in select]
axs.plot(x,y,label='Diversity 1 (FedAvg)', linestyle='--')

"""
Draw
"""
axs.set_xlabel('Communication round')
axs.set_ylabel('Global training loss')
axs.grid(True)

plt.title('Synthetic dataset - Loss / Communication round')
fig.tight_layout()
plt.legend()
plt.savefig('global_train_loss_diff_diversity.pdf')
plt.close()