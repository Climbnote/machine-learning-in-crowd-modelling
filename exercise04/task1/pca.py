import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, data):
        """parameter of the PCA

        :param data: given data
        :type data: numpy.ndarray
        """
        # perform Singular Value Decomposition(SVD)
        self.u, self.s, self.v_T = np.linalg.svd(data)
        # save the total number of Principal Components
        self.total_PCs = len(self.s)
        # depending on the dimnesion of data, diagonal matrix S is adjusted
        if (len(data) < len(data[0])):
            # set first dimension to bigger dimension
            self.first_dim = len(data[0])
             # generate real diagonal matrix s
            diag_matrix = np.diag(self.s)
            zero_matrix = np.zeros((self.total_PCs, self.first_dim - self.total_PCs), dtype=float)
            self.real_s = np.hstack((diag_matrix, zero_matrix))
        else:
            self.first_dim = len(data)
            # generate real diagonal matrix s
            diag_matrix = np.diag(self.s)
            zero_matrix = np.zeros((self.first_dim - self.total_PCs, self.total_PCs), dtype=float)
            self.real_s = np.concatenate((diag_matrix, zero_matrix))

        # calculate total energy of the data(Singular values)
        self.total_energy = np.square(self.s).sum()

    def perform_PCA(self, num_of_PC):
        """ perform principal component anaylsis on different number of principal components

        :param num_of_PC: number of principal components to be projected on
        :type num_of_PC: int
        :return energy_conserved: energy conserved during performing PCA
        :rtype energy_conserved: float
        return new_real_s: new modified S matrix after performing PCA
        rtype new_real_s: numpy.ndarray
        """
        # initialize modified energy conserved as total
        modified_energy = self.total_energy
        # initialize how many entries should be set to zero
        length_of_loop = self.total_PCs - num_of_PC
        new_real_s = np.copy(self.real_s)
        # generate new S matrix, by changing singular values to zero 
        for i in range(length_of_loop):
            modified_energy -= np.square(new_real_s[self.total_PCs-1-i][self.total_PCs-1-i])    
            new_real_s[self.total_PCs-1-i][self.total_PCs-1-i] = 0.0
        # calculate energy conserved
        energy_conserved = modified_energy / self.total_energy
        return energy_conserved, new_real_s

    def plot_projected_image_PCs(self, principal_components_values, real_s_for_PCA, energies_conserved):
        """ plot image after performing PCA
        
        :param principal_components_values: list of number of principal components to be plotted
        :type principal_components_values: class 'list'
        :param real_s_for_PCA: list of S matrices received after performing PCA
        :type real_s_for_PCA: class 'list'
        :param energies_conserved: list of energies conserved for number of pricinpal components used
        :type energies_conserved: class 'list'
        """
        for i in range(len(principal_components_values)):
            # recover new image
            new_image = (self.u.dot(real_s_for_PCA[i])).dot(self.v_T)
            # round energies conserved to three decimal places
            energies_conserved = np.round(energies_conserved, 3)
            # plot new image
            plt.title(f'Number of PCA Components: {principal_components_values[i]}, Energy Captured: {energies_conserved[i]}')
            plt.imshow(new_image, cmap='gray')
            plt.savefig('PCA_image_' + str(principal_components_values[i]) + '.pdf')
            plt.show()

    def plot_projected_path_PCs(self, principal_components_values, real_s_for_PCA, energies_conserved):
        """" plot paths after performing PCA on trajectories
        
        :param principal_components_values: list of number of principal components to be plotted
        :type principal_components_values: class 'list'
        :param real_s_for_PCA: list of S matrices received after performing PCA
        :type real_s_for_PCA: class 'list'
        :param energies_conserved: list of energies conserved for number of pricinpal components used
        :type energies_conserved: class 'list'
        """
        for i in range(len(principal_components_values)):
            # recover new path
            projected_trajectory = (self.u.dot(real_s_for_PCA[i])).dot(self.v_T)
            # get new values of X and Y coordinates for pedestrian
            X_of_pedestrians = np.zeros((15, 1000))
            Y_of_pedestrians = np.zeros((15, 1000))
            for j in range(15):
                X_of_pedestrians[j] = projected_trajectory[:, 2*j]
                Y_of_pedestrians[j] = projected_trajectory[:, 2*j + 1]
            # plot paths of two pedestrians with starting and ending points
            plt.scatter(X_of_pedestrians[0][0],Y_of_pedestrians[0][0], marker="o", c='red')
            plt.scatter(X_of_pedestrians[0][999],Y_of_pedestrians[0][999], marker="o", c='black')
            plt.plot(X_of_pedestrians[0], Y_of_pedestrians[0])
            plt.scatter(X_of_pedestrians[1][0],Y_of_pedestrians[1][0], marker="o", c='red')
            plt.scatter(X_of_pedestrians[1][999],Y_of_pedestrians[1][999], marker="o", c='black')
            plt.plot(X_of_pedestrians[1], Y_of_pedestrians[1])
            plt.title('Projection to the first two PCs, Energy captured: '+ str(np.round(energies_conserved[i], 3)))
            plt.savefig('Both_pedestrians_'+str(principal_components_values[i])+'_PCA.pdf')
            plt.show()

            

