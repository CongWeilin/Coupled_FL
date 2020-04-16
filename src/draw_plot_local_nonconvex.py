import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import smooth

fig, axs = plt.subplots()
STEP = 20
DATASET='mnist'
"""
Coupled_FL
"""
# diversity 0
with open('coupled_%s_noniid_0.pkl'%DATASET, 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(local_acc), step=STEP)
x = np.arange(len(select))
y = [local_acc[i] for i in select]
y = smooth(y)
axs.plot(x,y,label='Diversity 0 (Coupled_FL)', linestyle='-')

# diversity 2
with open('coupled_%s_noniid_2.pkl'%DATASET, 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(local_acc), step=STEP)
x = np.arange(len(select))
y = [local_acc[i] for i in select]
y = smooth(y)
axs.plot(x,y,label='Diversity 2 (Coupled_FL)', linestyle='-')

# # diversity 4
# with open('coupled_%s_noniid_4.pkl'%DATASET, 'rb') as f:
#     global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

# select = np.arange(len(local_acc), step=STEP)
# x = np.arange(len(select))
# y = [local_acc[i] for i in select]
# y = smooth(y)
# axs.plot(x,y,label='Diversity 4 (Coupled_FL)', linestyle='-')

"""
FedAvg
"""
# diversity 0
with open('fedavg_%s_noniid_0.pkl'%DATASET, 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(local_acc), step=STEP)
x = np.arange(len(select))
y = [local_acc[i] for i in select]
y = smooth(y)
axs.plot(x,y,label='Diversity 0 (FedAvg)', linestyle='--')

# diversity 2
with open('fedavg_%s_noniid_2.pkl'%DATASET, 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(local_acc), step=STEP)
x = np.arange(len(select))
y = [local_acc[i] for i in select]
y = smooth(y)
axs.plot(x,y,label='Diversity 2 (FedAvg)', linestyle='--')

# # diversity 4
# with open('fedavg_%s_noniid_4.pkl'%DATASET, 'rb') as f:
#     global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

# select = np.arange(len(local_acc), step=STEP)
# x = np.arange(len(select))
# y = [local_acc[i] for i in select]
# y = smooth(y)
# axs.plot(x,y,label='Diversity 4 (FedAvg)', linestyle='--')

"""
APFL
"""
# diversity 0
with open('apfl_%s_noniid_0.pkl'%DATASET, 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(local_acc), step=STEP)
x = np.arange(len(select))
y = [local_acc[i] for i in select]
y = smooth(y)
axs.plot(x,y,label='Diversity 0 (APFL)', linestyle='-')

# diversity 2
with open('apfl_%s_noniid_2.pkl'%DATASET, 'rb') as f:
    global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

select = np.arange(len(local_acc), step=STEP)
x = np.arange(len(select))
y = [local_acc[i] for i in select]
y = smooth(y)
axs.plot(x,y,label='Diversity 2 (APFL)', linestyle='-')

# # diversity 4
# with open('apfl_%s_noniid_4.pkl'%DATASET, 'rb') as f:
#     global_acc, global_loss, local_acc, local_loss, train_loss = pickle.load(f)

# select = np.arange(len(local_acc), step=STEP)
# x = np.arange(len(select))
# y = [local_acc[i] for i in select]
# y = smooth(y)
# axs.plot(x,y,label='Diversity 4 (APFL)', linestyle='-')



"""
Draw
"""
axs.set_xlabel('Communication round')
axs.set_ylabel('Local model accuracy')
axs.grid(True)

plt.title('Synthetic dataset - Accuracy / Communication round')
fig.tight_layout()
plt.legend()
plt.savefig('local_valid_acc_diff_diversity.pdf')
plt.close()