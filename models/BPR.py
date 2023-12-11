import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import scores


class BPR:
    user_count = 943
    item_count = 1682
    latent_factors = 20
    lr = 0.01
    reg = 0.01
    train_count = 1000
    train_data_path = 'train.txt'
    test_data_path = 'test.txt'
    size_u_i = user_count * item_count
    # 随机设定的U，V矩阵(即公式中的Wuk和Hik)矩阵
    U = np.random.rand(user_count, latent_factors) * 0.01 #大小无所谓
    V = np.random.rand(item_count, latent_factors) * 0.01
    biasV = np.random.rand(item_count) * 0.01

    test_data = np.zeros((user_count, item_count))

    test = np.zeros(size_u_i)

    predict_ = np.zeros(size_u_i)

    def load_data(self, path):
        user_ratings = defaultdict(set)
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i = line.split(" ")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split(' ')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1

    def train(self, user_ratings_train):
        for user in range(self.user_count):

            u = random.randint(1, self.user_count)
            if u not in user_ratings_train.keys():
                continue

            i = random.sample(user_ratings_train[u], 1)[0]

            j = random.randint(1, self.item_count)
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count)

            u = u - 1
            i = i - 1
            j = j - 1
            r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
            r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
            r_uij = r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_uij))
            # 更新2个矩阵
            self.U[u] += -self.lr * (loss_func * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (loss_func * self.U[u] + self.reg * self.V[i])
            self.V[j] += -self.lr * (loss_func * (-self.U[u]) + self.reg * self.V[j])
            # 更新偏置项
            self.biasV[i] += -self.lr * (loss_func + self.reg * self.biasV[i])
            self.biasV[j] += -self.lr * (-loss_func + self.reg * self.biasV[j])

    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    #主函数
    def main(self):
        # 获取U-I的{1:{2,5,1,2}....}数据
        user_ratings_train = self.load_data(self.train_data_path)
        # 获取测试集的评分矩阵
        self.load_test_data(self.test_data_path)
        # 将test_data矩阵拍平
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        # 训练
        for i in range(self.train_count):
            self.train(user_ratings_train)#训练1000次完成
        predict_matrix = self.predict(self.U, self.V) #将训练完成的矩阵內积
        # 预测
        self.predict_ = predict_matrix.getA().reshape(-1)  #.getA()将自身矩阵变量转化为ndarray类型的变量
        self.predict_ = pre_handle(user_ratings_train, self.predict_, self.item_count)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)
        # Top-K evaluation
        scores.topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count)

def pre_handle(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict


if __name__ == '__main__':
    # 调用类的主函数
    bpr = BPR()
    bpr.main()
