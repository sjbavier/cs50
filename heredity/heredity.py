import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    print(f'people: {people} one_gene: {one_gene} two_gene: {two_genes} have_trait: {have_trait} ')
    joint_prb = 1.0
    for person in people:
        gene_prob = 0.00
        trait_prob = 0.00

        person_trait = False
        if person in have_trait:
            person_trait = True

        person_genes = 0
        if person in one_gene:
            person_genes = 1
        elif person in two_genes:
            person_genes = 2
        else:
            person_genes = 0

        parent_prob = None
        if people[person]['father'] is not None and people[person]['mother'] is not None:
            def parent_gene_prob(parent):
                if parent in one_gene:
                    return 0.5
                elif parent in two_genes:
                    return 1 - PROBS['mutation']
                # parent has no genes return possible mutation probability
                else:
                    return PROBS['mutation']

            father_prob = parent_gene_prob(people[person]['father'])
            mother_prob = parent_gene_prob(people[person]['mother'])

            if person_genes == 2:
                # father passes and mother passes
                parent_prob = father_prob * mother_prob
            elif person_genes == 1:
                # sum of mother passing gene and father not and father passing gene and mother not
                parent_prob = (mother_prob * (1 - father_prob)) + (father_prob * (1 - mother_prob))
            else:
                # father doesn't pass gene and mother doesn't either
                parent_prob = (1 - father_prob) * (1 - mother_prob)

        # select gene probability
        if parent_prob is not None:
            gene_prob = parent_prob
        else:
            gene_prob = PROBS["gene"][person_genes]
        # select trait probability
        trait_prob = PROBS["trait"][person_genes][person_trait]

        # update joint_probability
        joint_prb *= (gene_prob * trait_prob)

    return joint_prb


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    print(f'probabilities: {probabilities}')
    for person in probabilities:
        person_trait = False
        if person in have_trait:
            person_trait = True

        person_genes = 0
        if person in one_gene:
            person_genes = 1
        elif person in two_genes:
            person_genes = 2
        else:
            person_genes = 0

        probabilities[person]["gene"][person_genes] = (probabilities[person]["gene"][person_genes] + p) / 2
        probabilities[person]["trait"][person_trait] = (probabilities[person]["trait"][person_trait] + p) / 2


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    print(f'probabilities: {probabilities}')

    for person in probabilities:
        total_probs = 0
        multiply_by = 0
        for gene in person["gene"]:
            total_probs += gene
        for trait in person["trait"]:
            total_probs += trait

        if total_probs <= 1:
            multiply_by = 1 / total_probs

        for i, gene in enumerate(["gene"]):
            probabilities[person]["gene"][i] = probabilities[person]["gene"][i] * multiply_by
        for trait in person["trait"]:
            total_probs += trait

if __name__ == "__main__":
    main()
