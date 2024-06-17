import itertools
import torch
from math import comb, factorial

def multinomial_coefficient(x1, x2, x3):
    # This gives the number of different orders for the combination (x1, x2, x3)
    return factorial(x1 + x2 + x3) // (factorial(x1) * factorial(x2) * factorial(x3))

def mendelian_prob(f, m, offspring):
    probs = torch.zeros(3, 3, 3)

    probs[0, 0, 0] = 1.0

    probs[0, 1, 0] = 0.5
    probs[0, 1, 1] = 0.5
    probs[1, 0, 0] = 0.5
    probs[1, 0, 1] = 0.5

    probs[0, 2, 1] = 1.0
    probs[2, 0, 1] = 1.0


    probs[1, 1, 0] = 0.25
    probs[1, 1, 1] = 0.5
    probs[1, 1, 2] = 0.25

    probs[2, 1, 1] = 0.5
    probs[2, 1, 2] = 0.5

    probs[1, 2, 1] = 0.5
    probs[1, 2, 2] = 0.5

    probs[2, 2, 2] = 1.0


    conf = [0, 0, 0]
    for geno in offspring:
        conf[geno] = conf[geno] + 1

    factor = multinomial_coefficient(conf[0], conf[1], conf[2])

    total_prob = 1.0
    for geno in offspring:
        total_prob *= probs[f, m, geno]

    return total_prob * factor


def enumerate_configurations(n, k):

    return [combo for combo in itertools.combinations_with_replacement(range(k), n)]


def combo_to_genotypes(combo, k):
    counts = [0] * k
    for item in combo:
        counts[item] += 1
    return counts

genotypes = [0, 1, 2]


class sufficient_statistic_single_variant:
    def __init__(self, father_genotype, mother_genotype, offspring_genotypes):

        self.n_offspring = sum(offspring_genotypes)
        self.father = father_genotype
        self.mother = mother_genotype
        self.offspring_genotypes = offspring_genotypes


        self.matrix = None
        self.number_of_configs = 0

        self.offspring_genotype_configs = None
        self.index = None

    def get_sufficient_stat(self):

        self.compute_matrix()
        geno_configs, probabilities = self.identify_suff_stat()

        return geno_configs, probabilities

    def compute_matrix(self):

        offspring_combinations = enumerate_configurations(n = self.n_offspring, k = 3)
        offspring_genotype_configs = [combo_to_genotypes(combo, 3) for combo in offspring_combinations]

        self.offspring_genotype_configs = offspring_genotype_configs

        # Number of possible offspring configurations
        self.number_of_configs = len(offspring_genotype_configs)

        # Create the matrix
        prob_tensor = torch.zeros((self.number_of_configs, 3, 3))

        for i, offspring_config in enumerate(offspring_genotype_configs):
            offspring_genos = []
            for j, count in enumerate(offspring_config):
                offspring_genos.extend([self.genotypes[j]] * count)
            for f in range(3):
                for m in range(3):
                    prob_tensor[i, f, m] = mendelian_prob(f, m, offspring_genos)

        if self.father is not None:
            prob_tensor[:, [i for i in range(3) if i != self.father], :] = 0

        if self.mother is not None:
            prob_tensor[:, :, [i for i in range(3) if i != self.mother]] = 0

        self.matrix = prob_tensor.reshape(self.number_of_configs, 9)
        self.index = self.offspring_genotype_configs.index(self.offspring_genotypes)


    def identify_suff_stat(self):

        row_norms = torch.norm(self.matrix, dim=1, keepdim=True)
        normalized_matrix = self.matrix / row_norms

        row_norm = torch.norm(self.matrix[self.index, :])
        normalized_row = self.matrix[self.index, :] / row_norm

        tmp_matrix = normalized_row.unsqueeze(0).expand_as(normalized_matrix)

        difference = torch.abs(normalized_matrix - tmp_matrix)

        geno_configs_suff_stat = torch.all(difference <= 0.00001, dim=1)

        sub_matrix = self.matrix[geno_configs_suff_stat, :].reshape(sum(geno_configs_suff_stat).item(), 9)
        compatible_mating_types = ~torch.all(sub_matrix == 0, dim=0)
        sub_matrix = sub_matrix[:, compatible_mating_types].reshape(sub_matrix.shape[0], sum(compatible_mating_types).item())
        probabilities = sub_matrix[:, 0] / sum(sub_matrix[:, 0])

        geno_configs = [self.offspring_genotype_configs[i] for i in torch.nonzero(geno_configs_suff_stat)]

        return geno_configs, probabilities



