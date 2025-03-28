import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        print(f'domains before: {self.domains}')
        for k, v in self.domains.items():
            for word in list(v.copy()):
                if len(word) != k.length:
                    v.remove(word)
                    print(f'removing word: {word}')
                    print(f'key: {k}, value: {v}')
        print(f'domains after: {self.domains}')

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.

        revised = False
        for x in x.domain:
            if no y in y.domain satisfies constraint for (x,y):
                delete x from x.domain
                revised = True

        """
        # instantiate as False
        revised = False

        overlaps = self.crossword.overlaps[x, y]
        if overlaps is None:
            return revised

        for xword in self.domains[x].copy():
            # check overlaps for arc consistency
            i, j = overlaps
            # if any word in y's domain that matches the letter
            if not any(xword[i] == yword[j] for yword in self.domains[y]):
                self.domains[x].remove(xword)
                revised = True
            revised = True

        print(f'overlaps {overlaps}')
        print(f'revise {x}, {y}')
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.

        queue = all arcs in csp
        while queue is not empty:
            (x, y) = Dequeue(queue)
            if Revise(csp, x y):
                if size of x.domain == 0:
                    return False
                for each Z in x.neighbors = {y}:
                    Enqueue(queue, (Z, X))
        return True
        """
        if arcs is None:
            arcs = []
            # create a list of arcs from all variables and their neighbors
            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    arcs.append((x, y))
        else:
            # ensure arcs is a list
            arcs = list(arcs)

        while arcs:
            # using a list to remove element at the end
            x, y = arcs.pop()

            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                # neighbors of x without y if it exists
                neighbors = self.crossword.neighbors(x).copy().discard(y)
                if neighbors is not None:
                    for z in neighbors:
                        # insert element at front of list for the backwards queue
                        arcs.insert(0, (z, x))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        print(f'assignment: {assignment}')
        for var in self.crossword.variables:
            if var not in assignment or assignment[var] is None:
                return False

        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        # check for correct variable length
        for var, value in assignment.items():
            for word in list(value):
                if len(word) != var.length:
                    return False

        # check for distinct values
        if (len(set(assignment.values())) != len(list(assignment.values()))):
            return False

        for var1 in assignment:
            for var2 in assignment:
                overlaps = self.crossword.overlaps[var1, var2]
                i, j = overlaps
                if var1[i] != var2[j]:
                    return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # TODO remove this:
        possible_values = set(self.domains[var])
        # eliminate values from var's values that are in assignment
        for word in self.domains[var]:
            if word not in set(assignment.values()):
                possible_values.add(word)

        def count_invalidated(possible_word):
            total = 0
            for neighbor in self.crossword.neighbors(var):
                # skip ones already assigned
                if neighbor in assignment:
                    continue
                for overlaps in self.crossword.overlaps[neighbor, var]:
                    (i, j) = overlaps
                    # skip ones without overlaps
                    if overlaps is not None:
                        for neighbor_word in self.domains[neighbor]:
                            if possible_word[i] != neighbor_word[j]:
                                total += 1
            return total

        return possible_values.sort(key=count_invalidated)

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        min_remaining = self.domains.copy()

        unassigned_options = [value for value in self.crossword.variables
                              if value not in assignment]

        def sort_min_remaining(var, domain):
            count = 0
            for value in domain not in assignment:
                count += 1
            min_remaining[var] = count

        sorted_min_remaining = sorted(min_remaining, key=sort_min_remaining)
        return self.domains[min_remaining_var]






    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        raise NotImplementedError


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
