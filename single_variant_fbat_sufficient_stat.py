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
    # Generate all possible combinations of non-negative integers that sum up to n
    return [combo for combo in itertools.combinations_with_replacement(range(k), n)]


# Convert combination to genotype counts
def combo_to_genotypes(combo, k):
    counts = [0] * k
    for item in combo:
        counts[item] += 1
    return counts




class sufficient_statistic_single_variant:
    def __init__(self, father_genotype, mother_genotype, offspring_genotypes):

        self.n_offspring = sum(offspring_genotypes)
        self.father = father_genotype
        self.mother = mother_genotype
        self.offspring_genotypes = offspring_genotypes

        self.genotypes = [0, 1, 2]
        self.matrix = None
        self.number_of_configs = 0

        self.offspring_genotype_configs = None
        self.index = None

    def get_sufficient_stat(self):

        self.compute_matrix()
        self.identify_suff_stat()

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



# pedigree CU0070F , nuclear family [A-CUHS-CU002788 x A-CUHS-CU002789]
# observed genotype configuration
# father         = 0
#                  0
# mother         = 0
#                  0
# offspring 1    = 1
#                  1
# offspring 2    = 1
#                  1
# offspring 3    = 1
#                  1
# offspring 4    = 1
#                  1
# offspring 5    = 1
#                  1
# offspring 6    = 2
#                  2
# offspring 7    = 1
#                  1
#
# compatible mating haplotype 1
# h2   :   1
# h1   :   2
#
# h2   :   1
# h1   :   2
#
# There are 3 compatible offspring genotypes:
#
# compatible offspring genotype 1 (g1) = 1/1
#
# compatible offspring genotype 2 (g2) = 1/2
#
# compatible offspring genotype 3 (g3) = 2/2
#
# distribution of compatible offspring genotype configurations:
#
# #g1     #g2     #g3     P[G]
# 5       1       1       0.007
# 4       2       1       0.035
# 4       1       2       0.017
# 3       3       1       0.092
# 3       2       2       0.069
# 3       1       3       0.023
# 2       4       1       0.138
# 2       3       2       0.138
# 2       2       3       0.069
# 2       1       4       0.017
# 1       5       1       0.111
# 1       4       2       0.138
# 1       3       3       0.092
# 1       2       4       0.035
# 1       1       5       0.007
# 6       0       1       0.001
# 5       0       2       0.002
# 4       0       3       0.003
# 3       0       4       0.003
# 2       0       5       0.002
# 1       0       6       0.001


ss = sufficient_statistic_single_variant(None, None, [6, 0, 1])
ss.compute_matrix()
g, p=ss.identify_suff_stat()

# pedigree CU0049F , nuclear family [A-CUHS-CU002159 x A-CUHS-CU002125]
# observed genotype configuration
# father         = 2
#                  2
# mother         = 2
#                  1
# offspring 1    = 2
#                  1
# offspring 2    = 2
#                  1
#
# compatible mating haplotype 1
# h1   :   2
# h1   :   2
#
# h1   :   2
# h2   :   1
#
# There are 2 compatible offspring genotypes:
#
# compatible offspring genotype 1 (g1) = 2/2
#
# compatible offspring genotype 2 (g2) = 2/1
#
# distribution of compatible offspring genotype configurations:
#
# #g1     #g2     P[G]
# 1       1       0.500
# 2       0       0.250
# 0       2       0.250

ss = sufficient_statistic_single_variant(2, 1, [0, 2, 0])
ss.compute_matrix()
g, p=ss.identify_suff_stat()




ss = sufficient_statistic_single_variant(None, None, [1, 1, 1])
ss.compute_matrix()
g, p=ss.identify_suff_stat()





