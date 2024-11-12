import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count
        self.safe = set()
        self.mine = set()

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # if the cells are the same length as the count
        # return those plus the known mines if any
        if len(self.cells) == self.count:
            for mine in self.cells:
                self.mine.add(mine)
        # return only the known mines
        return self.mine

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        # if the count of the sentence is 0 all cells are safe 
        # plus whatever has been marked
        if self.count == 0:
            # add to the known safes
            for safe in self.cells:
                self.safe.add(safe)
        # return the known safes
        return self.safe

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """

        # add marked mine to known mines
        self.mine.add(cell)
        if cell in self.cells:
            # remove count associated with cell set
            self.count -= 1
            # remove marked mine from set
            self.cells.remove(cell)

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if len(self.cells) == 0:
            return None

        # add cell to safe set
        self.safe.add(cell)
        # remove cell from cells set
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def enumerate_surrounding(self, cell):
        """
        returns set of surrounding cells within bounds
        """
        row_max = self.height - 1
        column_max = self.width - 1
        (row, col) = cell
        cell_set = set()

        for r in range(row - 1, row + 2):
            for c in range(col - 1, col + 2):
                if (r, c) == cell:
                    continue

                if 0 <= r <= row_max and 0 <= c <= column_max:
                    cell_set.add((r, c))

        return cell_set

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)

        # 2) mark the cell as safe
        self.mark_safe(cell)

        # 3) add a new sentence to the AI's knowledge base
        # based on the value of `cell` and `count`
        cell_set = self.enumerate_surrounding(cell) - self.mines - self.safes   # exclude known mines and safes
        new_sentence = Sentence(cells=cell_set, count=count)
        self.knowledge.append(new_sentence)

        # 4) mark any additional cells as safe or as mines
        # if it can be concluded based on the AI's knowledge base
        # for each sentence in ai knowledge check the known mines and add them to the ai
        changed = True
        while changed:
            changed = False
            safes = set()
            mines = set()

            # loop over sentences in knowledge and update
            for sentence in self.knowledge:
                safes.update(sentence.known_safes())
                mines.update(sentence.known_mines())


            if safes:
                for safe in safes.copy():
                    if safe not in self.safes:
                        self.mark_safe(safe)
                        changed = True

            if mines:
                for mine in mines.copy():
                    if mine not in self.mines:
                        self.mark_mine(mine)
                        changed = True

        # 5) add any new sentences to the AI's knowledge base
        # if they can be inferred from existing knowledge
        new_knowledge = []
        for sentence1 in self.knowledge.copy():
            for sentence2 in self.knowledge.copy():
                if sentence1 == sentence2:
                    continue
                if sentence1.cells.issubset(sentence2.cells):
                    derived_cells = sentence2.cells - sentence1.cells
                    derived_count = sentence2.count - sentence1.count
                    derived_sentence = Sentence(cells=derived_cells, count=derived_count)
                    if derived_sentence not in self.knowledge and derived_sentence not in new_knowledge:
                        new_knowledge.append(derived_sentence)
        self.knowledge.extend(new_knowledge)

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        not_explored_safes = self.safes.difference(self.moves_made)
        if len(not_explored_safes) != 0:
            next_cell = not_explored_safes.pop()
            self.moves_made.add(next_cell)
            return next_cell

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        cell_set = set()

        for i in range(self.height):
            for j in range(self.width):
                # if cell is mine or already moved skip
                if (i, j) in self.mines or (i, j) in self.moves_made:
                    continue

                cell_set.add((i, j))

        if not cell_set:
            return None
        return random.choice(list(cell_set))
