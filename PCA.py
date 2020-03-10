
# This python implementation of the PCA algorith has been done as pt_a joint work of Arpit Parwal (aparwal@usc.edu) and Yeon-soo Park (yeonsoop@usc.edu )
# The distances are provided in pt_a separate data file



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
class PrincipalComponentAnalysis():

    def __init__(self, datapoints=None, dimensions=0):
        self.x = datapoints
        self.k = dimensions
        self.main()

    def main(self):
        normalized_data = self.mean_normalization(self.x)
        cov_matrix = self.covariance_matrix(normalized_data)
        eigen_pairs = self.get_sorted_eigenpairs(cov_matrix, self.k)
        for i in eigen_pairs:
            print(i[1])

    def mean_normalization(self, x):
        #1 - mean = np.mean
        #2 - std = np.std(x, axis=0)

        return np.array((x - np.mean(x, axis=0)))

    def covariance_matrix(self, x):
        covariance = np.cov(x.T)
        #sn.heatmap(covariance, annot=True, fmt='g')
        return covariance

    def get_sorted_eigenpairs(self, covarience, k):

        eig_vals, eig_vecs = np.linalg.eig(covarience)
        #print(eig_vals, eig_vecs)
        #eig_vecs, eig_vals, _v = np.linalg.svd(covarience)
        #print(eig_vals, eig_vecs)
        #_u, _s, _v = np.linalg.svd(covarience)

        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        #print('Eigenvalues in descending order:')
        #for i in eig_pairs:
        #    print(i[0])

        self.plot_explained_variances(eig_vals)
        return [eig_pairs[i] for i in range(k)]

    def plot_explained_variances(self, eig_vals):
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(4, 3))

            plt.bar(range(len(eig_vals)), var_exp, alpha=0.5, align='center',
                    label='individual explained variance')
            #plt.step(range(len(eig_vals)), cum_var_exp, where='mid',
            #         label='cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal components')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

    def plot3D(self, data):

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        for i in range(len(data)):
            x, y, z = data[i][0], data[i][1], data[i][2]
            plotted = ax.scatter(x, y, z, c='b', marker='o')

        ax.legend([plotted], ['data'])
        plt.show()


def getInputData(filename):
    data = np.genfromtxt(filename, delimiter='\t')
    return data


if __name__ == '__main__':

    dimension = 2
    datapoints = np.array(getInputData('pca-data.txt'))
    PCA = PrincipalComponentAnalysis(datapoints, dimension)
