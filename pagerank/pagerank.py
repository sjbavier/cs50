import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.

    The corpus is a Python dictionary mapping a page name to a set of all pages linked to by that page.
    The page is a string representing which page the random surfer is currently on.
    The damping_factor is a floating point number representing the damping factor to be used when generating the probabilities.

    With probability damping_factor, the random surfer should randomly choose one of the links from page with equal probability.
    With probability 1 - damping_factor, the random surfer should randomly choose one of all pages in the corpus with equal probability.
    """

    # first case, no outbound links all pages have equal probability
    links = corpus[page]
    prob_distribution = {}
    len_corpus = len(corpus)
    if len(links) == 0:
        eq_probability = 1 / len_corpus
        for k, _v in corpus.items():
            prob_distribution[k] = eq_probability
        return prob_distribution

    # divide probability of 0.85 amongst linked pages
    dampening_factor_choices = damping_factor / len(corpus[page])

    # divide remainder percentages amongst all pages 0.15
    remainder_dampening = (1 - damping_factor) / len_corpus #

    # distribute probabilities
    for k, _v in corpus.items():

        # looping through the corpus if key in the current pages
        if k in corpus[page]:
            prob_distribution[k] = dampening_factor_choices + remainder_dampening
            continue
        # else just add the dampening remainder
        prob_distribution[k] = remainder_dampening

    return prob_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    count = 0
    page_frequency = {key: 0 for key in corpus.keys()} # copy all keys from corpus and set all values to 0
    random_page = ""

    while count < n:
        
        # on first pass, choose random page
        if count == 0:
            # random choice
            random_page = random.choice(list(corpus.keys()))
            # add page as key and number of times accessed as count, in the end we will divide by the num of samples
            page_frequency[random_page] += 1
            count += 1
            continue
        
        prob_dist = transition_model(corpus=corpus, page=random_page, damping_factor=damping_factor)
        page_keys = list(prob_dist.keys())
        page_values = list(prob_dist.values())
        random_page = np.random.choice(keys=page_keys, p=page_values)
        page_frequency[random_page] += 1
        count += 1
    
    samples = {}
    for key, value in page_frequency:
        samples[key] = value / n # aggregation of values should be n
    
    return samples



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    len_corpus = len( corpus )
    page_ranks = {}
    damping_const = 1 - damping_factor / len(corpus)
    convergence = False
    threshold = .001
    
    while not convergence:
        for page in corpus:
            for second_page in corpus:
                if page in second_page:
                    

    




if __name__ == "__main__":
    main()
